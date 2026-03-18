/**
 * WASM-accelerated element-wise logaddexp.
 *
 * Binary: out[i] = log(exp(a[i]) + exp(b[i]))  (same-shape contiguous arrays)
 * Scalar: out[i] = log(exp(a[i]) + exp(scalar))
 * Returns null if WASM can't handle this case.
 * Float types use native kernels; int64/uint64 use native i64 kernels
 * that convert to f64 in WASM (avoiding JS BigInt→Number overhead).
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
import { ensureMemory, resetAllocator, copyIn, alloc, copyOut } from './runtime';
import { ArrayStorage } from '../storage';
import { promoteDTypes, type DType, type TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 64;

type BinaryFn = (aPtr: number, bPtr: number, outPtr: number, N: number) => void;
type ScalarFn = (aPtr: number, outPtr: number, N: number, scalar: number) => void;

const binaryKernels: Partial<Record<DType, BinaryFn>> = {
  float64: logaddexp_f64,
  float32: logaddexp_f32,
};

const scalarKernels: Partial<Record<DType, ScalarFn>> = {
  float64: logaddexp_scalar_f64,
  float32: logaddexp_scalar_f32,
};

const intBinaryKernels: Partial<Record<DType, BinaryFn>> = {
  int64: logaddexp_i64,
  uint64: logaddexp_u64,
  int32: logaddexp_i32,
  uint32: logaddexp_u32,
  int16: logaddexp_i16,
  uint16: logaddexp_u16,
  int8: logaddexp_i8,
  uint8: logaddexp_u8,
};

const intScalarKernels: Partial<Record<DType, ScalarFn>> = {
  int64: logaddexp_scalar_i64,
  uint64: logaddexp_scalar_u64,
  int32: logaddexp_scalar_i32,
  uint32: logaddexp_scalar_u32,
  int16: logaddexp_scalar_i16,
  uint16: logaddexp_scalar_u16,
  int8: logaddexp_scalar_i8,
  uint8: logaddexp_scalar_u8,
};

type AnyTypedArrayCtor = new (length: number) => TypedArray;
const ctorMap: Partial<Record<DType, AnyTypedArrayCtor>> = {
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
 * WASM-accelerated element-wise logaddexp of two same-shape contiguous arrays.
 * Returns null if WASM can't handle.
 */
export function wasmLogaddexp(a: ArrayStorage, b: ArrayStorage): ArrayStorage | null {
  if (!a.isCContiguous || !b.isCContiguous) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = promoteDTypes(a.dtype, b.dtype);

  // Float path
  const floatKernel = binaryKernels[dtype];
  if (floatKernel) {
    const Ctor = ctorMap[dtype]!;
    const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;

    ensureMemory(size * bpe * 3);
    resetAllocator();

    const aData = a.data.subarray(a.offset, a.offset + size) as TypedArray;
    const bData = b.data.subarray(b.offset, b.offset + size) as TypedArray;

    const aPtr = copyIn(aData);
    const bPtr = copyIn(bData);
    const outPtr = alloc(size * bpe);

    floatKernel(aPtr, bPtr, outPtr, size);

    const outData = copyOut(
      outPtr,
      size,
      Ctor as unknown as new (buffer: ArrayBuffer, byteOffset: number, length: number) => TypedArray
    );
    return ArrayStorage.fromData(outData, Array.from(a.shape), dtype);
  }

  // Int64 path
  const intKernel = intBinaryKernels[dtype];
  if (intKernel) {
    const InputCtor = ctorMap[dtype]!;
    const inputBpe = (InputCtor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
    const outBpe = 8;

    ensureMemory(size * inputBpe * 2 + size * outBpe);
    resetAllocator();

    const aData = a.data.subarray(a.offset, a.offset + size) as TypedArray;
    const bData = b.data.subarray(b.offset, b.offset + size) as TypedArray;

    const aPtr = copyIn(aData);
    const bPtr = copyIn(bData);
    const outPtr = alloc(size * outBpe);

    intKernel(aPtr, bPtr, outPtr, size);

    const outData = copyOut(
      outPtr,
      size,
      Float64Array as unknown as new (
        buffer: ArrayBuffer,
        byteOffset: number,
        length: number
      ) => TypedArray
    );
    return ArrayStorage.fromData(outData, Array.from(a.shape), 'float64');
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

  const dtype = a.dtype;

  // Float path
  const floatKernel = scalarKernels[dtype];
  if (floatKernel) {
    const Ctor = ctorMap[dtype]!;
    const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;

    ensureMemory(size * bpe * 2);
    resetAllocator();

    const aData = a.data.subarray(a.offset, a.offset + size) as TypedArray;
    const aPtr = copyIn(aData);
    const outPtr = alloc(size * bpe);

    floatKernel(aPtr, outPtr, size, scalar);

    const outData = copyOut(
      outPtr,
      size,
      Ctor as unknown as new (buffer: ArrayBuffer, byteOffset: number, length: number) => TypedArray
    );
    return ArrayStorage.fromData(outData, Array.from(a.shape), dtype);
  }

  // Int64 path
  const intKernel = intScalarKernels[dtype];
  if (intKernel) {
    const InputCtor = ctorMap[dtype]!;
    const inputBpe = (InputCtor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
    const outBpe = 8;

    ensureMemory(size * inputBpe + size * outBpe);
    resetAllocator();

    const aData = a.data.subarray(a.offset, a.offset + size) as TypedArray;
    const aPtr = copyIn(aData);
    const outPtr = alloc(size * outBpe);

    intKernel(aPtr, outPtr, size, scalar);

    const outData = copyOut(
      outPtr,
      size,
      Float64Array as unknown as new (
        buffer: ArrayBuffer,
        byteOffset: number,
        length: number
      ) => TypedArray
    );
    return ArrayStorage.fromData(outData, Array.from(a.shape), 'float64');
  }

  return null;
}
