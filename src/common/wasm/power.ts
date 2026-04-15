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
import {
  wasmMalloc,
  resetScratchAllocator,
  resolveInputPtr,
  scratchCopyIn,
  f16InputToScratchF32,
  f32OutputToF16Region,
} from './runtime';
import { ArrayStorage } from '../storage';
import {
  effectiveDType,
  promoteDTypes,
  isBigIntDType,
  type DType,
  type TypedArray,
} from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 32;

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
  float16: power_f32,
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
  float16: power_scalar_f32,
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
 * WASM-accelerated element-wise power of two same-shape contiguous arrays.
 * Returns null if WASM can't handle.
 */
export function wasmPower(a: ArrayStorage, b: ArrayStorage): ArrayStorage | null {
  if (!a.isCContiguous || !b.isCContiguous) return null;
  // WASM kernels expect same-dtype inputs — bail on mixed dtypes
  if (a.dtype !== b.dtype) return null;
  // Must be same size — WASM kernel doesn't broadcast
  if (a.size !== b.size) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = effectiveDType(promoteDTypes(a.dtype, b.dtype));
  const kernel = binaryKernels[dtype];
  const Ctor = ctorMap[dtype];
  if (!kernel || !Ctor) return null;

  const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const outBytes = size * bpe;

  const outRegion = wasmMalloc(outBytes);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;

  const originalDtype = dtype;
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

  kernel(aPtr, bPtr, outRegion.ptr, size);

  if (originalDtype === 'float16') {
    const f16Region = f32OutputToF16Region(outRegion, size);
    outRegion.release();
    if (!f16Region) return null;
    return ArrayStorage.fromWasmRegion(
      Array.from(a.shape),
      originalDtype,
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

/**
 * WASM-accelerated element-wise power with scalar exponent.
 * Returns null if WASM can't handle.
 */
export function wasmPowerScalar(a: ArrayStorage, scalar: number): ArrayStorage | null {
  if (!a.isCContiguous) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = effectiveDType(a.dtype);

  // Integer types with negative or non-integer exponents need float promotion
  // (matches NumPy behavior: int ** negative -> float64).
  // Convert to float64 and use the f64 kernel.
  const isIntegerType = dtype !== 'float32' && dtype !== 'float64';
  if (isIntegerType && (scalar < 0 || !Number.isInteger(scalar))) {
    const bpe = 8; // f64
    const outBytes = size * bpe;

    const outRegion = wasmMalloc(outBytes);
    if (!outRegion) return null;

    wasmConfig.wasmCallCount++;

    resetScratchAllocator();
    const aOff = a.offset;
    const src = a.data;
    const converted = new Float64Array(size);
    if (isBigIntDType(dtype)) {
      for (let i = 0; i < size; i++) converted[i] = Number(src[aOff + i]!);
    } else {
      for (let i = 0; i < size; i++) converted[i] = src[aOff + i] as number;
    }

    const aPtr = scratchCopyIn(converted as unknown as TypedArray);

    power_scalar_f64(aPtr, outRegion.ptr, size, scalar);

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

  const kernel = scalarKernels[dtype];
  const Ctor = ctorMap[dtype];
  if (!kernel || !Ctor) return null;

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

  kernel(aPtr, outRegion.ptr, size, scalar);

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
