/**
 * WASM-accelerated element-wise power.
 *
 * Binary: out[i] = a[i] ^ b[i]  (same-shape contiguous arrays)
 * Scalar: out[i] = a[i] ^ scalar
 * Returns null if WASM can't handle this case.
 */

import {
  power_f64,
  power_f32,
  power_i64,
  power_i32,
  power_i16,
  power_i8,
  power_scalar_f64,
  power_scalar_f32,
  power_scalar_i64,
  power_scalar_i32,
  power_scalar_i16,
  power_scalar_i8,
} from './bins/power.wasm';
import { ensureMemory, resetAllocator, copyIn, alloc, copyOut } from './runtime';
import { ArrayStorage } from '../storage';
import { promoteDTypes, isBigIntDType, type DType, type TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 64;

type BinaryFn = (aPtr: number, bPtr: number, outPtr: number, N: number) => void;
type ScalarFn = (aPtr: number, outPtr: number, N: number, scalar: number) => void;

const binaryKernels: Partial<Record<DType, BinaryFn>> = {
  float64: power_f64,
  float32: power_f32,
  int64: power_i64,
  uint64: power_i64,
  int32: power_i32,
  uint32: power_i32,
  int16: power_i16,
  uint16: power_i16,
  int8: power_i8,
  uint8: power_i8,
};

const scalarKernels: Partial<Record<DType, ScalarFn>> = {
  float64: power_scalar_f64,
  float32: power_scalar_f32,
  int64: power_scalar_i64,
  uint64: power_scalar_i64,
  int32: power_scalar_i32,
  uint32: power_scalar_i32,
  int16: power_scalar_i16,
  uint16: power_scalar_i16,
  int8: power_scalar_i8,
  uint8: power_scalar_i8,
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
 * WASM-accelerated element-wise power of two same-shape contiguous arrays.
 * Returns null if WASM can't handle.
 */
export function wasmPower(a: ArrayStorage, b: ArrayStorage): ArrayStorage | null {
  if (!a.isCContiguous || !b.isCContiguous) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = promoteDTypes(a.dtype, b.dtype);
  const kernel = binaryKernels[dtype];
  const Ctor = ctorMap[dtype];
  if (!kernel || !Ctor) return null;

  const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const aBytes = size * bpe;
  const bBytes = size * bpe;
  const outBytes = size * bpe;

  ensureMemory(aBytes + bBytes + outBytes);
  resetAllocator();

  const aOff = a.offset;
  const bOff = b.offset;
  const aData = a.data.subarray(aOff, aOff + size) as TypedArray;
  const bData = b.data.subarray(bOff, bOff + size) as TypedArray;

  const aPtr = copyIn(aData);
  const bPtr = copyIn(bData);
  const outPtr = alloc(outBytes);

  kernel(aPtr, bPtr, outPtr, size);

  const outData = copyOut(
    outPtr,
    size,
    Ctor as unknown as new (buffer: ArrayBuffer, byteOffset: number, length: number) => TypedArray
  );

  return ArrayStorage.fromData(outData, Array.from(a.shape), dtype);
}

/**
 * WASM-accelerated element-wise power with scalar exponent.
 * Returns null if WASM can't handle.
 */
export function wasmPowerScalar(a: ArrayStorage, scalar: number): ArrayStorage | null {
  if (!a.isCContiguous) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = a.dtype;

  // Integer types with negative or non-integer exponents need float promotion
  // (matches NumPy behavior: int ** negative → float64).
  // Convert to float64 and use the f64 kernel.
  const isIntegerType = dtype !== 'float32' && dtype !== 'float64';
  if (isIntegerType && (scalar < 0 || !Number.isInteger(scalar))) {
    const bpe = 8; // f64
    ensureMemory(size * bpe * 2);
    resetAllocator();

    const aOff = a.offset;
    const src = a.data;
    const converted = new Float64Array(size);
    if (isBigIntDType(dtype)) {
      for (let i = 0; i < size; i++) converted[i] = Number(src[aOff + i]!);
    } else {
      for (let i = 0; i < size; i++) converted[i] = src[aOff + i] as number;
    }

    const aPtr = copyIn(converted as unknown as TypedArray);
    const outPtr = alloc(size * bpe);

    power_scalar_f64(aPtr, outPtr, size, scalar);

    const outData = copyOut(
      outPtr,
      size,
      Float64Array as unknown as new (buffer: ArrayBuffer, byteOffset: number, length: number) => TypedArray
    );
    return ArrayStorage.fromData(outData, Array.from(a.shape), 'float64');
  }

  const kernel = scalarKernels[dtype];
  const Ctor = ctorMap[dtype];
  if (!kernel || !Ctor) return null;

  const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const aBytes = size * bpe;
  const outBytes = size * bpe;

  ensureMemory(aBytes + outBytes);
  resetAllocator();

  const aOff = a.offset;
  const aData = a.data.subarray(aOff, aOff + size) as TypedArray;

  const aPtr = copyIn(aData);
  const outPtr = alloc(outBytes);

  kernel(aPtr, outPtr, size, scalar);

  const outData = copyOut(
    outPtr,
    size,
    Ctor as unknown as new (buffer: ArrayBuffer, byteOffset: number, length: number) => TypedArray
  );

  return ArrayStorage.fromData(outData, Array.from(a.shape), dtype);
}
