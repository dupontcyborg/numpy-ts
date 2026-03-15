/**
 * WASM-accelerated element-wise division.
 *
 * Binary: out[i] = a[i] / b[i]
 * Scalar: out[i] = a[i] / scalar
 * Returns null if WASM can't handle this case.
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
import { ensureMemory, resetAllocator, copyIn, alloc, copyOut } from './runtime';
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
  complex128: div_c128,
  complex64: div_c64,
};

// Float scalar kernels (same-dtype in, same-dtype out)
const scalarKernels: Partial<Record<DType, ScalarFn>> = {
  float64: div_scalar_f64,
  float32: div_scalar_f32,
};

// Integer-to-f64 binary kernels (int/uint in, f64 out)
const intBinaryKernels: Partial<Record<DType, BinaryFn>> = {
  int64: div_i64_f64, uint64: div_u64_f64,
  int32: div_i32_f64, uint32: div_u32_f64,
  int16: div_i16_f64, uint16: div_u16_f64,
  int8: div_i8_f64,  uint8: div_u8_f64,
};

// Integer-to-f64 scalar kernels (int/uint in, f64 out)
const intScalarKernels: Partial<Record<DType, ScalarFn>> = {
  int64: div_scalar_i64_f64, uint64: div_scalar_u64_f64,
  int32: div_scalar_i32_f64, uint32: div_scalar_u32_f64,
  int16: div_scalar_i16_f64, uint16: div_scalar_u16_f64,
  int8: div_scalar_i8_f64,  uint8: div_scalar_u8_f64,
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
    ensureMemory(dataLen * bpe * 3);
    resetAllocator();

    const aPtr = copyIn(
      a.data.subarray(a.offset * complexFactor, (a.offset + size) * complexFactor) as TypedArray
    );
    const bPtr = copyIn(
      b.data.subarray(b.offset * complexFactor, (b.offset + size) * complexFactor) as TypedArray
    );
    const outPtr = alloc(dataLen * bpe);
    kernel(aPtr, bPtr, outPtr, size);

    const outData = copyOut(
      outPtr,
      dataLen,
      Ctor as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
    );
    return ArrayStorage.fromData(outData, Array.from(a.shape), dtype);
  }

  // Try integer-to-f64 kernel
  const intKernel = intBinaryKernels[dtype];
  const InCtor = ctorMap[dtype];
  if (!intKernel || !InCtor) return null;

  const inBpe = (InCtor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  ensureMemory(size * inBpe * 2 + size * 8);
  resetAllocator();

  const aPtr = copyIn(a.data.subarray(a.offset, a.offset + size) as TypedArray);
  const bPtr = copyIn(b.data.subarray(b.offset, b.offset + size) as TypedArray);
  const outPtr = alloc(size * 8);
  intKernel(aPtr, bPtr, outPtr, size);

  const outData = copyOut(
    outPtr,
    size,
    Float64Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
  );
  return ArrayStorage.fromData(outData, Array.from(a.shape), 'float64');
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
    ensureMemory(size * bpe * 2);
    resetAllocator();

    const aPtr = copyIn(a.data.subarray(a.offset, a.offset + size) as TypedArray);
    const outPtr = alloc(size * bpe);
    kernel(aPtr, outPtr, size, scalar);

    const outData = copyOut(
      outPtr,
      size,
      Ctor as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
    );
    return ArrayStorage.fromData(outData, Array.from(a.shape), dtype);
  }

  // Try integer-to-f64 scalar kernel
  const intKernel = intScalarKernels[dtype];
  const InCtor = ctorMap[dtype];
  if (!intKernel || !InCtor) return null;

  const inBpe = (InCtor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  ensureMemory(size * inBpe + size * 8);
  resetAllocator();

  const aPtr = copyIn(a.data.subarray(a.offset, a.offset + size) as TypedArray);
  const outPtr = alloc(size * 8);
  intKernel(aPtr, outPtr, size, scalar);

  const outData = copyOut(
    outPtr,
    size,
    Float64Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
  );
  return ArrayStorage.fromData(outData, Array.from(a.shape), 'float64');
}
