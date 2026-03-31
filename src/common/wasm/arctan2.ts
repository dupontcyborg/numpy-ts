/**
 * WASM-accelerated element-wise arctan2.
 *
 * Binary: out[i] = atan2(a[i], b[i])  (same-shape contiguous arrays)
 * Returns null if WASM can't handle this case.
 */

import { arctan2_f64, arctan2_f32 } from './bins/arctan2.wasm';
import {
  wasmMalloc,
  resetScratchAllocator,
  resolveInputPtr,
  scratchCopyIn,
  copyOut,
  f16ToF32Input,
  f32ToF16Output,
} from './runtime';
import { ArrayStorage } from '../storage';
import { promoteDTypes, type DType, type TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 64;

type BinaryFn = (aPtr: number, bPtr: number, outPtr: number, N: number) => void;

const binaryKernels: Partial<Record<DType, BinaryFn>> = {
  float64: arctan2_f64,
  float32: arctan2_f32,
  float16: arctan2_f32,
};

type AnyTypedArrayCtor = new (length: number) => TypedArray;
const ctorMap: Partial<Record<DType, AnyTypedArrayCtor>> = {
  float64: Float64Array,
  float32: Float32Array,
  float16: Float32Array,
};

/**
 * WASM-accelerated element-wise arctan2 of two same-shape contiguous arrays.
 * Returns null if WASM can't handle.
 */
export function wasmArctan2(a: ArrayStorage, b: ArrayStorage): ArrayStorage | null {
  if (!a.isCContiguous || !b.isCContiguous) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = promoteDTypes(a.dtype, b.dtype);

  const floatKernel = binaryKernels[dtype];
  if (!floatKernel) return null;

  const Ctor = ctorMap[dtype]!;
  const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const outBytes = size * bpe;

  const outRegion = wasmMalloc(outBytes);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;

  resetScratchAllocator();

  let aPtr: number;
  let bPtr: number;
  if (dtype === 'float16') {
    const aData = f16ToF32Input(a.data.subarray(a.offset, a.offset + size) as TypedArray, a.dtype);
    const bData = f16ToF32Input(b.data.subarray(b.offset, b.offset + size) as TypedArray, b.dtype);
    aPtr = scratchCopyIn(aData);
    bPtr = scratchCopyIn(bData);
  } else {
    aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, bpe);
    bPtr = resolveInputPtr(b.data, b.isWasmBacked, b.wasmPtr, b.offset, size, bpe);
  }

  floatKernel(aPtr, bPtr, outRegion.ptr, size);

  if (dtype === 'float16') {
    const outData = copyOut(
      outRegion.ptr,
      size,
      Ctor as unknown as new (buffer: ArrayBuffer, byteOffset: number, length: number) => TypedArray
    );
    outRegion.release();
    const finalOut = f32ToF16Output(outData, dtype);
    return ArrayStorage.fromData(finalOut, Array.from(a.shape), dtype);
  }

  return ArrayStorage.fromWasmRegion(
    Array.from(a.shape),
    dtype,
    outRegion,
    size,
    Ctor as unknown as new (buffer: ArrayBuffer, byteOffset: number, length: number) => TypedArray
  );
}
