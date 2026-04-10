/**
 * WASM-accelerated element-wise logaddexp.
 *
 * Binary: out[i] = log(exp(a[i]) + exp(b[i]))  (same-shape contiguous arrays)
 * Scalar: out[i] = log(exp(a[i]) + exp(scalar))
 * Returns null if WASM can't handle this case.
 * Float types use native kernels; integer types use type-appropriate output:
 *   i8/u8 → f32 (then downcast to f16 if available)
 *   i16/u16 → f32
 *   i32/u32/i64/u64 → f64
 */

import {
  logaddexp_f64,
  logaddexp_f32,
  logaddexp_scalar_f64,
  logaddexp_scalar_f32,
  logaddexp_i64,
  logaddexp_scalar_i64,
  logaddexp_i32,
  logaddexp_scalar_i32,
  logaddexp_i16,
  logaddexp_scalar_i16,
  logaddexp_i8,
  logaddexp_scalar_i8,
  logaddexp_u64,
  logaddexp_scalar_u64,
  logaddexp_u32,
  logaddexp_scalar_u32,
  logaddexp_u16,
  logaddexp_scalar_u16,
  logaddexp_u8,
  logaddexp_scalar_u8,
} from './bins/logaddexp.wasm';
import {
  wasmMalloc,
  resetScratchAllocator,
  resolveInputPtr,
  f16InputToScratchF32,
  f32OutputToF16Region,
} from './runtime';
import { ArrayStorage } from '../storage';
import { effectiveDType, hasFloat16, promoteDTypes, type DType, type TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 64;

type BinaryFn = (aPtr: number, bPtr: number, outPtr: number, N: number) => void;
type ScalarFn = (aPtr: number, outPtr: number, N: number, scalar: number) => void;

const binaryKernels: Partial<Record<DType, BinaryFn>> = {
  float64: logaddexp_f64,
  float32: logaddexp_f32,
  float16: logaddexp_f32,
};

const scalarKernels: Partial<Record<DType, ScalarFn>> = {
  float64: logaddexp_scalar_f64,
  float32: logaddexp_scalar_f32,
  float16: logaddexp_scalar_f32,
};

// Large int → f64 output (i32/u32/i64/u64 need f64 precision)
const largeIntBinaryKernels: Partial<Record<DType, BinaryFn>> = {
  int64: logaddexp_i64,
  uint64: logaddexp_u64,
  int32: logaddexp_i32,
  uint32: logaddexp_u32,
};

// Small int → f32 output (i8/u8/i16/u16 → f32, then optionally downcast to f16)
const smallIntBinaryKernels: Partial<Record<DType, BinaryFn>> = {
  int16: logaddexp_i16,
  uint16: logaddexp_u16,
  int8: logaddexp_i8,
  uint8: logaddexp_u8,
};

const largeIntScalarKernels: Partial<Record<DType, ScalarFn>> = {
  int64: logaddexp_scalar_i64,
  uint64: logaddexp_scalar_u64,
  int32: logaddexp_scalar_i32,
  uint32: logaddexp_scalar_u32,
};

const smallIntScalarKernels: Partial<Record<DType, ScalarFn>> = {
  int16: logaddexp_scalar_i16,
  uint16: logaddexp_scalar_u16,
  int8: logaddexp_scalar_i8,
  uint8: logaddexp_scalar_u8,
};

type AnyTypedArrayCtor = new (length: number) => TypedArray;
const ctorMap: Partial<Record<DType, AnyTypedArrayCtor>> = {
  float64: Float64Array,
  float32: Float32Array,
  float16: Float32Array,
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
 * WASM-accelerated element-wise logaddexp of two same-shape contiguous arrays.
 * Returns null if WASM can't handle.
 */
export function wasmLogaddexp(a: ArrayStorage, b: ArrayStorage): ArrayStorage | null {
  if (!a.isCContiguous || !b.isCContiguous) return null;

  // WASM kernels expect same-dtype inputs — bail on mixed dtypes
  if (a.dtype !== b.dtype) return null;

  // WASM kernel does not broadcast — sizes must match
  if (a.size !== b.size) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = effectiveDType(promoteDTypes(a.dtype, b.dtype));

  // Float path
  const floatKernel = binaryKernels[dtype];
  if (floatKernel) {
    const Ctor = ctorMap[dtype]!;
    const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
    const outBytes = size * bpe;

    const outRegion = wasmMalloc(outBytes);
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
      Ctor as unknown as new (buffer: ArrayBuffer, byteOffset: number, length: number) => TypedArray
    );
  }

  // Small int path: i8/u8/i16/u16 → f32 output, optionally downcast to f16
  const smallKernel = smallIntBinaryKernels[dtype];
  if (smallKernel) {
    const InputCtor = ctorMap[dtype]!;
    const inputBpe = (InputCtor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
    const outBytes = size * 4; // f32
    const outRegion = wasmMalloc(outBytes);
    if (!outRegion) return null;

    wasmConfig.wasmCallCount++;
    resetScratchAllocator();
    const aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, inputBpe);
    const bPtr = resolveInputPtr(b.data, b.isWasmBacked, b.wasmPtr, b.offset, size, inputBpe);

    smallKernel(aPtr, bPtr, outRegion.ptr, size);

    if (hasFloat16 && (dtype === 'int8' || dtype === 'uint8')) {
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
  const largeKernel = largeIntBinaryKernels[dtype];
  if (largeKernel) {
    const InputCtor = ctorMap[dtype]!;
    const inputBpe = (InputCtor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
    const outBytes = size * 8;

    const outRegion = wasmMalloc(outBytes);
    if (!outRegion) return null;

    wasmConfig.wasmCallCount++;

    resetScratchAllocator();
    const aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, inputBpe);
    const bPtr = resolveInputPtr(b.data, b.isWasmBacked, b.wasmPtr, b.offset, size, inputBpe);

    largeKernel(aPtr, bPtr, outRegion.ptr, size);

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

/**
 * WASM-accelerated element-wise logaddexp with scalar.
 * Returns null if WASM can't handle.
 */
export function wasmLogaddexpScalar(a: ArrayStorage, scalar: number): ArrayStorage | null {
  if (!a.isCContiguous) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = effectiveDType(a.dtype);

  // Float path
  const floatKernel = scalarKernels[dtype];
  if (floatKernel) {
    const Ctor = ctorMap[dtype]!;
    const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
    const outBytes = size * bpe;

    const outRegion = wasmMalloc(outBytes);
    if (!outRegion) return null;

    wasmConfig.wasmCallCount++;

    resetScratchAllocator();

    let aPtr: number;
    if (dtype === 'float16') {
      aPtr = f16InputToScratchF32(a, size);
    } else {
      aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, bpe);
    }

    floatKernel(aPtr, outRegion.ptr, size, scalar);

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
      Ctor as unknown as new (buffer: ArrayBuffer, byteOffset: number, length: number) => TypedArray
    );
  }

  // Small int path: i8/u8/i16/u16 → f32 output, optionally downcast to f16
  const smallKernel = smallIntScalarKernels[dtype];
  if (smallKernel) {
    const InputCtor = ctorMap[dtype]!;
    const inputBpe = (InputCtor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
    const outBytes = size * 4; // f32
    const outRegion = wasmMalloc(outBytes);
    if (!outRegion) return null;

    wasmConfig.wasmCallCount++;
    resetScratchAllocator();
    const aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, inputBpe);

    smallKernel(aPtr, outRegion.ptr, size, scalar);

    if (hasFloat16 && (dtype === 'int8' || dtype === 'uint8')) {
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
  const largeKernel = largeIntScalarKernels[dtype];
  if (largeKernel) {
    const InputCtor = ctorMap[dtype]!;
    const inputBpe = (InputCtor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
    const outBytes = size * 8;

    const outRegion = wasmMalloc(outBytes);
    if (!outRegion) return null;

    wasmConfig.wasmCallCount++;

    resetScratchAllocator();
    const aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, inputBpe);

    largeKernel(aPtr, outRegion.ptr, size, scalar);

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
