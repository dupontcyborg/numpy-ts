/**
 * WASM-accelerated element-wise base-2 exponential.
 *
 * Unary: out[i] = exp2(a[i])
 * Returns null if WASM can't handle this case.
 * Float types use native kernels; integer types are converted to float64
 * in JS and run through the f64 SIMD kernel (matches NumPy's promotion).
 */

import { exp2_f64, exp2_f32, exp2_i64, exp2_u64 } from './bins/exp2.wasm';
import { ensureMemory, resetAllocator, copyIn, alloc, copyOut } from './runtime';
import { ArrayStorage } from '../storage';
import { isComplexDType, isBigIntDType, type DType, type TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 64;

type UnaryFn = (aPtr: number, outPtr: number, N: number) => void;

const kernels: Partial<Record<DType, UnaryFn>> = {
  float64: exp2_f64,
  float32: exp2_f32,
};

/**
 * WASM-accelerated element-wise base-2 exponential.
 * Returns null if WASM can't handle (complex, non-contiguous, too small).
 */
export function wasmExp2(a: ArrayStorage): ArrayStorage | null {
  if (!a.isCContiguous) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = a.dtype;
  if (isComplexDType(dtype)) return null;

  // Native float path
  const nativeKernel = kernels[dtype];
  if (nativeKernel) {
    const isF32 = dtype === 'float32';
    const bpe = isF32 ? 4 : 8;
    const Ctor = isF32 ? Float32Array : Float64Array;

    ensureMemory(size * bpe * 2);
    resetAllocator();

    const aOff = a.offset;
    const aData = a.data.subarray(aOff, aOff + size) as TypedArray;
    const aPtr = copyIn(aData);
    const outPtr = alloc(size * bpe);

    nativeKernel(aPtr, outPtr, size);

    const outData = copyOut(
      outPtr,
      size,
      Ctor as unknown as new (buffer: ArrayBuffer, byteOffset: number, length: number) => TypedArray
    );
    return ArrayStorage.fromData(outData, Array.from(a.shape), dtype);
  }

  // int64/uint64 native path — avoid costly BigInt→Number conversion
  if (dtype === 'int64' || dtype === 'uint64') {
    ensureMemory(size * 16); // 8 bytes in + 8 bytes out
    resetAllocator();
    const aOff = a.offset;
    const aData = a.data.subarray(aOff, aOff + size) as TypedArray;
    const aPtr = copyIn(aData);
    const outPtr = alloc(size * 8);
    (dtype === 'int64' ? exp2_i64 : exp2_u64)(aPtr, outPtr, size);
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

  // Integer path: convert to float64, run SIMD f64 kernel
  // (NumPy promotes int→float64 for exp2)
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

  exp2_f64(aPtr, outPtr, size);

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
