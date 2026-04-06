/**
 * WASM-accelerated element-wise heaviside step function.
 *
 * Scalar: out[i] = x1[i] < 0 ? 0 : x1[i] == 0 ? x2 : 1
 * Binary: out[i] = x1[i] < 0 ? 0 : x1[i] == 0 ? x2[i] : 1
 * Returns null if WASM can't handle this case.
 */

import {
  heaviside_scalar_f64,
  heaviside_scalar_f32,
  heaviside_f64,
  heaviside_f32,
} from './bins/heaviside.wasm';
import {
  wasmMalloc,
  resetScratchAllocator,
  resolveInputPtr,
  f16InputToScratchF32,
  f32OutputToF16Region,
} from './runtime';
import { ArrayStorage } from '../storage';
import { effectiveDType, type DType, type TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 64;

type ScalarFn = (x1Ptr: number, outPtr: number, N: number, x2: number) => void;
type BinaryFn = (x1Ptr: number, x2Ptr: number, outPtr: number, N: number) => void;

const scalarKernels: Partial<Record<DType, ScalarFn>> = {
  float64: heaviside_scalar_f64,
  float32: heaviside_scalar_f32,
  float16: heaviside_scalar_f32,
};

const binaryKernels: Partial<Record<DType, BinaryFn>> = {
  float64: heaviside_f64,
  float32: heaviside_f32,
  float16: heaviside_f32,
};

type AnyTypedArrayCtor = new (length: number) => TypedArray;
const ctorMap: Partial<Record<DType, AnyTypedArrayCtor>> = {
  float64: Float64Array,
  float32: Float32Array,
  float16: Float32Array,
};

export function wasmHeavisideScalar(
  x1: ArrayStorage,
  x2: number,
  resultDtype: 'float64' | 'float32' | 'float16'
): ArrayStorage | null {
  if (!x1.isCContiguous) return null;
  const dtype = effectiveDType(resultDtype) as 'float64' | 'float32' | 'float16';
  const size = x1.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const kernel = scalarKernels[dtype];
  const Ctor = ctorMap[dtype];
  if (!kernel || !Ctor) return null;

  const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;

  const outRegion = wasmMalloc(size * bpe);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  if (dtype === 'float16') {
    const x1Ptr = f16InputToScratchF32(x1, size);
    kernel(x1Ptr, outRegion.ptr, size, x2);

    const f16Region = f32OutputToF16Region(outRegion, size);
    outRegion.release();
    if (!f16Region) return null;
    return ArrayStorage.fromWasmRegion(
      Array.from(x1.shape),
      resultDtype,
      f16Region,
      size,
      Float16Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
    );
  }

  const x1Ptr = resolveInputPtr(x1.data, x1.isWasmBacked, x1.wasmPtr, x1.offset, size, bpe);
  kernel(x1Ptr, outRegion.ptr, size, x2);

  return ArrayStorage.fromWasmRegion(
    Array.from(x1.shape),
    resultDtype,
    outRegion,
    size,
    Ctor as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
  );
}

export function wasmHeaviside(
  x1: ArrayStorage,
  x2: ArrayStorage,
  resultDtype: 'float64' | 'float32' | 'float16'
): ArrayStorage | null {
  if (!x1.isCContiguous || !x2.isCContiguous) return null;
  const dtype = effectiveDType(resultDtype) as 'float64' | 'float32' | 'float16';
  const size = x1.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const kernel = binaryKernels[dtype];
  const Ctor = ctorMap[dtype];
  if (!kernel || !Ctor) return null;

  const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;

  const outRegion = wasmMalloc(size * bpe);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  if (dtype === 'float16') {
    const x1Ptr = f16InputToScratchF32(x1, size);
    const x2Ptr = f16InputToScratchF32(x2, size);
    kernel(x1Ptr, x2Ptr, outRegion.ptr, size);

    const f16Region = f32OutputToF16Region(outRegion, size);
    outRegion.release();
    if (!f16Region) return null;
    return ArrayStorage.fromWasmRegion(
      Array.from(x1.shape),
      resultDtype,
      f16Region,
      size,
      Float16Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
    );
  }

  const x1Ptr = resolveInputPtr(x1.data, x1.isWasmBacked, x1.wasmPtr, x1.offset, size, bpe);
  const x2Ptr = resolveInputPtr(x2.data, x2.isWasmBacked, x2.wasmPtr, x2.offset, size, bpe);
  kernel(x1Ptr, x2Ptr, outRegion.ptr, size);

  return ArrayStorage.fromWasmRegion(
    Array.from(x1.shape),
    resultDtype,
    outRegion,
    size,
    Ctor as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
  );
}
