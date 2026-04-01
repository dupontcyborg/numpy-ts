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
  f16InputToScratchF32,
  f32OutputToF16Region,
} from './runtime';
import { ArrayStorage } from '../storage';
import {
  promoteDTypes,
  isComplexDType,
  isBigIntDType,
  type DType,
  type TypedArray,
} from '../dtype';
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

  // WASM kernel does not broadcast — sizes must match
  if (a.size !== b.size) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = promoteDTypes(a.dtype, b.dtype);
  if (isComplexDType(dtype)) return null;

  const floatKernel = binaryKernels[dtype];

  // Float path (f64, f32)
  if (floatKernel) {
    const Ctor = ctorMap[dtype]!;
    const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;

    const outRegion = wasmMalloc(size * bpe);
    if (!outRegion) return null;

    wasmConfig.wasmCallCount++;
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

    floatKernel(aPtr, bPtr, outRegion.ptr, size);

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
      Ctor as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
    );
  }

  // Integer path: convert both inputs to f64, use f64 kernel (NumPy promotes int→float64)
  const outRegion = wasmMalloc(size * 8);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  const aOff = a.offset;
  const bOff = b.offset;
  const aSrc = a.data;
  const bSrc = b.data;
  const aF64 = new Float64Array(size);
  const bF64 = new Float64Array(size);
  if (isBigIntDType(a.dtype)) {
    for (let i = 0; i < size; i++) aF64[i] = Number(aSrc[aOff + i]!);
  } else {
    for (let i = 0; i < size; i++) aF64[i] = aSrc[aOff + i] as number;
  }
  if (isBigIntDType(b.dtype)) {
    for (let i = 0; i < size; i++) bF64[i] = Number(bSrc[bOff + i]!);
  } else {
    for (let i = 0; i < size; i++) bF64[i] = bSrc[bOff + i] as number;
  }

  const aPtr = scratchCopyIn(aF64 as unknown as TypedArray);
  const bPtr = scratchCopyIn(bF64 as unknown as TypedArray);

  arctan2_f64(aPtr, bPtr, outRegion.ptr, size);

  return ArrayStorage.fromWasmRegion(
    Array.from(a.shape),
    'float64',
    outRegion,
    size,
    Float64Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
  );
}
