/**
 * WASM-accelerated element-wise division (float types only).
 *
 * Binary: out[i] = a[i] / b[i]
 * Scalar: out[i] = a[i] / scalar
 * Returns null if WASM can't handle this case.
 */

import {
  div_f64, div_f32,
  div_scalar_f64, div_scalar_f32,
} from './bins/divide.wasm';
import { ensureMemory, resetAllocator, copyIn, alloc, copyOut } from './runtime';
import { ArrayStorage } from '../storage';
import type { DType, TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 64;

type BinaryFn = (aPtr: number, bPtr: number, outPtr: number, N: number) => void;
type ScalarFn = (aPtr: number, outPtr: number, N: number, scalar: number) => void;

const binaryKernels: Partial<Record<DType, BinaryFn>> = {
  float64: div_f64, float32: div_f32,
};

const scalarKernels: Partial<Record<DType, ScalarFn>> = {
  float64: div_scalar_f64, float32: div_scalar_f32,
};

type AnyTypedArrayCtor = new (length: number) => TypedArray;
const ctorMap: Partial<Record<DType, AnyTypedArrayCtor>> = {
  float64: Float64Array, float32: Float32Array,
};

export function wasmDiv(a: ArrayStorage, b: ArrayStorage): ArrayStorage | null {
  if (!a.isCContiguous || !b.isCContiguous) return null;
  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  // Only handle same-dtype float arrays
  if (a.dtype !== b.dtype) return null;
  const dtype = a.dtype;
  const kernel = binaryKernels[dtype];
  const Ctor = ctorMap[dtype];
  if (!kernel || !Ctor) return null;

  const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  ensureMemory(size * bpe * 3);
  resetAllocator();

  const aPtr = copyIn(a.data.subarray(a.offset, a.offset + size) as TypedArray);
  const bPtr = copyIn(b.data.subarray(b.offset, b.offset + size) as TypedArray);
  const outPtr = alloc(size * bpe);
  kernel(aPtr, bPtr, outPtr, size);

  const outData = copyOut(outPtr, size, Ctor as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray);
  return ArrayStorage.fromData(outData, Array.from(a.shape), dtype);
}

export function wasmDivScalar(a: ArrayStorage, scalar: number): ArrayStorage | null {
  if (!a.isCContiguous) return null;
  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = a.dtype;
  const kernel = scalarKernels[dtype];
  const Ctor = ctorMap[dtype];
  if (!kernel || !Ctor) return null;

  const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  ensureMemory(size * bpe * 2);
  resetAllocator();

  const aPtr = copyIn(a.data.subarray(a.offset, a.offset + size) as TypedArray);
  const outPtr = alloc(size * bpe);
  kernel(aPtr, outPtr, size, scalar);

  const outData = copyOut(outPtr, size, Ctor as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray);
  return ArrayStorage.fromData(outData, Array.from(a.shape), dtype);
}
