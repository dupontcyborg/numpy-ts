/**
 * WASM-accelerated element-wise hypot.
 *
 * Binary: out[i] = hypot(a[i], b[i])  (same-shape contiguous arrays)
 * Scalar: out[i] = hypot(a[i], scalar)
 * Returns null if WASM can't handle this case.
 * Float types use native kernels; int64/uint64 use native i64 kernels
 * that convert to f64 in WASM (avoiding JS BigInt->Number overhead).
 */

import {
  hypot_f64,
  hypot_f32,
  hypot_scalar_f64,
  hypot_scalar_f32,
  hypot_i64,
  hypot_scalar_i64,
  hypot_i32,
  hypot_scalar_i32,
  hypot_i16,
  hypot_scalar_i16,
  hypot_i8,
  hypot_scalar_i8,
} from './bins/hypot.wasm';
import {
  wasmMalloc,
  resetScratchAllocator,
  resolveInputPtr,
  f16InputToScratchF32,
  f32OutputToF16Region,
} from './runtime';
import { ArrayStorage } from '../storage';
import { promoteDTypes, type DType, type TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 64;

type BinaryFn = (aPtr: number, bPtr: number, outPtr: number, N: number) => void;
type ScalarFn = (aPtr: number, outPtr: number, N: number, scalar: number) => void;

const binaryKernels: Partial<Record<DType, BinaryFn>> = {
  float64: hypot_f64,
  float32: hypot_f32,
  float16: hypot_f32,
};

const scalarKernels: Partial<Record<DType, ScalarFn>> = {
  float64: hypot_scalar_f64,
  float32: hypot_scalar_f32,
  float16: hypot_scalar_f32,
};

// Int64 kernels: native int input, f64 output
const intBinaryKernels: Partial<Record<DType, BinaryFn>> = {
  int64: hypot_i64,
  uint64: hypot_i64,
  int32: hypot_i32,
  uint32: hypot_i32,
  int16: hypot_i16,
  uint16: hypot_i16,
  int8: hypot_i8,
  uint8: hypot_i8,
};

const intScalarKernels: Partial<Record<DType, ScalarFn>> = {
  int64: hypot_scalar_i64,
  uint64: hypot_scalar_i64,
  int32: hypot_scalar_i32,
  uint32: hypot_scalar_i32,
  int16: hypot_scalar_i16,
  uint16: hypot_scalar_i16,
  int8: hypot_scalar_i8,
  uint8: hypot_scalar_i8,
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
 * WASM-accelerated element-wise hypot of two same-shape contiguous arrays.
 * Returns null if WASM can't handle.
 */
export function wasmHypot(a: ArrayStorage, b: ArrayStorage): ArrayStorage | null {
  if (!a.isCContiguous || !b.isCContiguous) return null;

  // WASM kernel does not broadcast — sizes must match
  if (a.size !== b.size) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = promoteDTypes(a.dtype, b.dtype);

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

  // Int path: native int input, f64 output
  const intKernel = intBinaryKernels[dtype];
  if (intKernel) {
    const InputCtor = ctorMap[dtype]!;
    const inputBpe = (InputCtor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
    const outBytes = size * 8;

    const outRegion = wasmMalloc(outBytes);
    if (!outRegion) return null;

    wasmConfig.wasmCallCount++;

    resetScratchAllocator();
    const aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, inputBpe);
    const bPtr = resolveInputPtr(b.data, b.isWasmBacked, b.wasmPtr, b.offset, size, inputBpe);

    intKernel(aPtr, bPtr, outRegion.ptr, size);

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
 * WASM-accelerated element-wise hypot with scalar.
 * Returns null if WASM can't handle.
 */
export function wasmHypotScalar(a: ArrayStorage, scalar: number): ArrayStorage | null {
  if (!a.isCContiguous) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = a.dtype;

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

  // Int path
  const intKernel = intScalarKernels[dtype];
  if (intKernel) {
    const InputCtor = ctorMap[dtype]!;
    const inputBpe = (InputCtor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
    const outBytes = size * 8;

    const outRegion = wasmMalloc(outBytes);
    if (!outRegion) return null;

    wasmConfig.wasmCallCount++;

    resetScratchAllocator();
    const aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, inputBpe);

    intKernel(aPtr, outRegion.ptr, size, scalar);

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
