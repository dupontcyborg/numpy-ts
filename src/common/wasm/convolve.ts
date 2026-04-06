/**
 * WASM-accelerated 1D convolution (full mode).
 *
 * Computes full linear convolution, returns float64 or float32 result.
 * The JS ops layer handles mode slicing (same/valid) and complex types.
 * Returns null if WASM can't handle this case.
 */

import { convolve_f64, convolve_f32 } from './bins/convolve.wasm';
import { wasmMalloc, resetScratchAllocator, scratchCopyIn, f16InputToScratchF32 } from './runtime';
import { ArrayStorage } from '../storage';
import { effectiveDType, type DType, TypedArray } from '../dtype';
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

  // Use float64 unless both are float32 (or float16, which uses f32 kernel)
  const aDtype = effectiveDType(a.dtype);
  const vDtype = effectiveDType(v.dtype);
  const bothF32Like =
    (aDtype === 'float32' || aDtype === 'float16') &&
    (vDtype === 'float32' || vDtype === 'float16');
  const dtype: DType = bothF32Like ? 'float32' : 'float64';
  const kernel = kernels[dtype];
  const Ctor = ctorMap[dtype];
  if (!kernel || !Ctor) return null;

  const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const outBytes = outLen * bpe;

  const outRegion = wasmMalloc(outBytes);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  const aOff = a.offset;
  const vOff = v.offset;

  // Convert to target float type if input dtype differs (e.g. int32 -> float64)
  let aPtr: number;
  let vPtr: number;
  if (aDtype === 'float16') {
    aPtr = f16InputToScratchF32(a, aLen);
  } else if (aDtype === dtype) {
    aPtr = scratchCopyIn(a.data.subarray(aOff, aOff + aLen) as TypedArray);
  } else {
    const tmp = new (Ctor as unknown as new (len: number) => TypedArray)(aLen);
    const src = a.data;
    for (let i = 0; i < aLen; i++) tmp[i] = Number(src[aOff + i]!);
    aPtr = scratchCopyIn(tmp);
  }
  if (vDtype === 'float16') {
    vPtr = f16InputToScratchF32(v, vLen);
  } else if (vDtype === dtype) {
    vPtr = scratchCopyIn(v.data.subarray(vOff, vOff + vLen) as TypedArray);
  } else {
    const tmp = new (Ctor as unknown as new (len: number) => TypedArray)(vLen);
    const src = v.data;
    for (let i = 0; i < vLen; i++) tmp[i] = Number(src[vOff + i]!);
    vPtr = scratchCopyIn(tmp);
  }

  kernel(aPtr, aLen, vPtr, vLen, outRegion.ptr, outLen);

  return ArrayStorage.fromWasmRegion(
    [outLen],
    dtype,
    outRegion,
    outLen,
    Ctor as unknown as new (buffer: ArrayBuffer, byteOffset: number, length: number) => TypedArray
  );
}
