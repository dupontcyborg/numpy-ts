/**
 * WASM-accelerated 1D convolution (full mode).
 *
 * Computes full linear convolution, returns float64 or float32 result.
 * The JS ops layer handles mode slicing (same/valid) and complex types.
 * Returns null if WASM can't handle this case.
 */

import { convolve_f64, convolve_f32 } from './bins/convolve.wasm';
import { ensureMemory, resetAllocator, copyIn, alloc, copyOut } from './runtime';
import { ArrayStorage } from '../storage';
import type { DType, TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 32;

type ConvolveFn = (
  aPtr: number,
  aLen: number,
  vPtr: number,
  vLen: number,
  outPtr: number,
  outLen: number
) => void;

const kernels: Partial<Record<DType, ConvolveFn>> = {
  float64: convolve_f64,
  float32: convolve_f32,
};

type AnyTypedArrayCtor = new (length: number) => TypedArray;
const ctorMap: Partial<Record<DType, AnyTypedArrayCtor>> = {
  float64: Float64Array,
  float32: Float32Array,
};

/**
 * WASM-accelerated 1D convolution (full mode).
 * Returns null if WASM can't handle (complex, non-contiguous, too small).
 */
export function wasmConvolve(a: ArrayStorage, v: ArrayStorage): ArrayStorage | null {
  if (!a.isCContiguous || !v.isCContiguous) return null;

  const aLen = a.size;
  const vLen = v.size;
  const outLen = aLen + vLen - 1;

  if (outLen < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  // Use float64 unless both are float32
  const dtype: DType = a.dtype === 'float32' && v.dtype === 'float32' ? 'float32' : 'float64';
  const kernel = kernels[dtype];
  const Ctor = ctorMap[dtype];
  if (!kernel || !Ctor) return null;

  const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const aBytes = aLen * bpe;
  const vBytes = vLen * bpe;
  const outBytes = outLen * bpe;

  ensureMemory(aBytes + vBytes + outBytes);
  resetAllocator();

  const aOff = a.offset;
  const vOff = v.offset;

  // Convert to target float type if input dtype differs (e.g. int32 -> float64)
  let aData: TypedArray;
  let vData: TypedArray;
  if (a.dtype === dtype) {
    aData = a.data.subarray(aOff, aOff + aLen) as TypedArray;
  } else {
    const tmp = new (Ctor as unknown as new (len: number) => TypedArray)(aLen);
    const src = a.data;
    for (let i = 0; i < aLen; i++) tmp[i] = Number(src[aOff + i]!);
    aData = tmp;
  }
  if (v.dtype === dtype) {
    vData = v.data.subarray(vOff, vOff + vLen) as TypedArray;
  } else {
    const tmp = new (Ctor as unknown as new (len: number) => TypedArray)(vLen);
    const src = v.data;
    for (let i = 0; i < vLen; i++) tmp[i] = Number(src[vOff + i]!);
    vData = tmp;
  }

  const aPtr = copyIn(aData);
  const vPtr = copyIn(vData);
  const outPtr = alloc(outBytes);

  kernel(aPtr, aLen, vPtr, vLen, outPtr, outLen);

  const outData = copyOut(
    outPtr,
    outLen,
    Ctor as unknown as new (buffer: ArrayBuffer, byteOffset: number, length: number) => TypedArray
  );

  return ArrayStorage.fromData(outData, [outLen], dtype);
}
