/**
 * WASM-accelerated element-wise GCD (greatest common divisor).
 *
 * Scalar: out[i] = gcd(a[i], scalar)
 * Binary: out[i] = gcd(a[i], b[i])
 * Preserves the promoted integer dtype (NumPy behavior).
 * Returns null if WASM can't handle this case.
 */

import {
  gcd_scalar_i32,
  gcd_i32,
  gcd_i16,
  gcd_u16,
  gcd_i8,
  gcd_u8,
  gcd_scalar_i16,
  gcd_scalar_u16,
  gcd_scalar_i8,
  gcd_scalar_u8,
} from './bins/gcd.wasm';
import { wasmMalloc, resetScratchAllocator, resolveInputPtr } from './runtime';
import { ArrayStorage } from '../storage';
import { promoteDTypes, getTypedArrayConstructor, type DType, type TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 32;

type BinaryFn = (aPtr: number, bPtr: number, outPtr: number, N: number) => void;
type ScalarFn = (aPtr: number, outPtr: number, N: number, scalar: number) => void;

const binaryKernels: Partial<Record<DType, BinaryFn>> = {
  int32: gcd_i32,
  int16: gcd_i16,
  uint16: gcd_u16,
  int8: gcd_i8,
  uint8: gcd_u8,
};

const scalarKernels: Partial<Record<DType, ScalarFn>> = {
  int32: gcd_scalar_i32,
  int16: gcd_scalar_i16,
  uint16: gcd_scalar_u16,
  int8: gcd_scalar_i8,
  uint8: gcd_scalar_u8,
};

const bpeMap: Partial<Record<DType, number>> = {
  int32: 4,
  int16: 2,
  uint16: 2,
  int8: 1,
  uint8: 1,
};

export function wasmGcdScalar(a: ArrayStorage, scalar: number): ArrayStorage | null {
  if (!a.isCContiguous) return null;
  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = a.dtype as DType;
  const kernel = scalarKernels[dtype];
  const bpe = bpeMap[dtype];
  if (!kernel || !bpe) return null;

  const outRegion = wasmMalloc(size * bpe);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  const aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, bpe);
  kernel(aPtr, outRegion.ptr, size, Math.abs(Math.trunc(scalar)));

  const Ctor = getTypedArrayConstructor(dtype)!;
  return ArrayStorage.fromWasmRegion(
    Array.from(a.shape),
    dtype,
    outRegion,
    size,
    Ctor as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
  );
}

export function wasmGcd(a: ArrayStorage, b: ArrayStorage): ArrayStorage | null {
  if (!a.isCContiguous || !b.isCContiguous) return null;
  // WASM kernels expect same-dtype inputs — bail on mixed dtypes
  if (a.dtype !== b.dtype) return null;
  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const outDtype = promoteDTypes(a.dtype, b.dtype);
  const kernel = binaryKernels[outDtype];
  const bpe = bpeMap[outDtype];
  if (!kernel || !bpe) return null;

  const outRegion = wasmMalloc(size * bpe);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  const aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, bpe);
  const bPtr = resolveInputPtr(b.data, b.isWasmBacked, b.wasmPtr, b.offset, size, bpe);
  kernel(aPtr, bPtr, outRegion.ptr, size);

  const Ctor = getTypedArrayConstructor(outDtype)!;
  return ArrayStorage.fromWasmRegion(
    Array.from(a.shape),
    outDtype,
    outRegion,
    size,
    Ctor as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
  );
}
