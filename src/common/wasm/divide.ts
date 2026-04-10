/**
 * WASM-accelerated element-wise division.
 *
 * Binary: out[i] = a[i] / b[i]
 * Scalar: out[i] = a[i] / scalar
 * Returns null if WASM can't handle this case.
 * Integer types use type-appropriate output:
 *   i8/u8 → f32 (then downcast to f16 if available)
 *   i16/u16 → f32
 *   i32/u32/i64/u64 → f64
 */

import {
  div_f64,
  div_f32,
  div_scalar_f64,
  div_scalar_f32,
  div_i64_f64,
  div_scalar_i64_f64,
  div_i32_f64,
  div_scalar_i32_f64,
  div_i16_f64,
  div_scalar_i16_f64,
  div_i8_f64,
  div_scalar_i8_f64,
  div_u64_f64,
  div_scalar_u64_f64,
  div_u32_f64,
  div_scalar_u32_f64,
  div_u16_f64,
  div_scalar_u16_f64,
  div_u8_f64,
  div_scalar_u8_f64,
  div_c128,
  div_c64,
} from './bins/divide.wasm';
import { wasmMalloc, resetScratchAllocator, resolveInputPtr } from './runtime';
import { ArrayStorage } from '../storage';
import { isComplexDType, type DType, type TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 64;

type BinaryFn = (aPtr: number, bPtr: number, outPtr: number, N: number) => void;
type ScalarFn = (aPtr: number, outPtr: number, N: number, scalar: number) => void;

// Float/complex binary kernels (same-dtype in, same-dtype out)
const binaryKernels: Partial<Record<DType, BinaryFn>> = {
  float64: div_f64,
  float32: div_f32,
  // float16 excluded: f16→f32 conversion overhead makes JS path faster
  complex128: div_c128,
  complex64: div_c64,
};

// Float scalar kernels (same-dtype in, same-dtype out)
const scalarKernels: Partial<Record<DType, ScalarFn>> = {
  float64: div_scalar_f64,
  float32: div_scalar_f32,
  // float16 excluded: f16→f32 conversion overhead makes JS path faster
};

// All int → f64 binary kernels (NumPy: all int divide → f64)
const intBinaryKernels: Partial<Record<DType, BinaryFn>> = {
  int64: div_i64_f64,
  uint64: div_u64_f64,
  int32: div_i32_f64,
  uint32: div_u32_f64,
  int16: div_i16_f64,
  uint16: div_u16_f64,
  int8: div_i8_f64,
  uint8: div_u8_f64,
};

// All int → f64 scalar kernels
const intScalarKernels: Partial<Record<DType, ScalarFn>> = {
  int64: div_scalar_i64_f64,
  uint64: div_scalar_u64_f64,
  int32: div_scalar_i32_f64,
  uint32: div_scalar_u32_f64,
  int16: div_scalar_i16_f64,
  uint16: div_scalar_u16_f64,
  int8: div_scalar_i8_f64,
  uint8: div_scalar_u8_f64,
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
  complex128: Float64Array,
  complex64: Float32Array,
  int64: BigInt64Array,
  uint64: BigUint64Array,
  int32: Int32Array,
  uint32: Uint32Array,
  int16: Int16Array,
  uint16: Uint16Array,
  int8: Int8Array,
  uint8: Uint8Array,
};

export function wasmDiv(a: ArrayStorage, b: ArrayStorage): ArrayStorage | null {
  if (!a.isCContiguous || !b.isCContiguous) return null;
  // WASM kernel does not broadcast — sizes must match
  if (a.size !== b.size) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  // Only handle same-dtype arrays
  if (a.dtype !== b.dtype) return null;
  const dtype = a.dtype;

  // Try float/complex kernel first
  const kernel = binaryKernels[dtype];
  if (kernel) {
    const Ctor = ctorMap[dtype]!;
    const complexFactor = isComplexDType(dtype) ? 2 : 1;
    const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
    const dataLen = size * complexFactor;
    const outBytes = dataLen * bpe;

    const outRegion = wasmMalloc(outBytes);
    if (!outRegion) return null;

    wasmConfig.wasmCallCount++;

    resetScratchAllocator();
    const aPtr = resolveInputPtr(
      a.data,
      a.isWasmBacked,
      a.wasmPtr,
      a.offset * complexFactor,
      dataLen,
      bpe
    );
    const bPtr = resolveInputPtr(
      b.data,
      b.isWasmBacked,
      b.wasmPtr,
      b.offset * complexFactor,
      dataLen,
      bpe
    );

    kernel(aPtr, bPtr, outRegion.ptr, size);

    return ArrayStorage.fromWasmRegion(
      Array.from(a.shape),
      dtype,
      outRegion,
      dataLen,
      Ctor as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
    );
  }

  const inBpe = bpeMap[dtype];
  if (!inBpe) return null;

  // All int → f64 output (NumPy: all int divide → f64)
  const intKernel = intBinaryKernels[dtype];
  if (!intKernel) return null;

  const outBytes = size * 8;
  const outRegion = wasmMalloc(outBytes);
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

export function wasmDivScalar(a: ArrayStorage, scalar: number): ArrayStorage | null {
  if (!a.isCContiguous) return null;
  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = a.dtype;

  // Try float scalar kernel first
  const kernel = scalarKernels[dtype];
  if (kernel) {
    const Ctor = ctorMap[dtype]!;
    const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
    const outBytes = size * bpe;

    const outRegion = wasmMalloc(outBytes);
    if (!outRegion) return null;

    wasmConfig.wasmCallCount++;

    resetScratchAllocator();
    const aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, bpe);

    kernel(aPtr, outRegion.ptr, size, scalar);

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

  // All int → f64 scalar path
  const intKernel = intScalarKernels[dtype];
  if (!intKernel) return null;

  const outBytes = size * 8;
  const outRegion = wasmMalloc(outBytes);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();
  const aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, inBpe);

  intKernel(aPtr, outRegion.ptr, size, scalar);

  return ArrayStorage.fromWasmRegion(
    Array.from(a.shape),
    'float64',
    outRegion,
    size,
    Float64Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
  );
}
