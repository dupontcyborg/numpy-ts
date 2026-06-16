/**
 * WASM-accelerated element-wise logaddexp2: log2(2^a + 2^b).
 *
 * Float-only fast path (f64/f32/f16); integer & complex inputs return null so the
 * caller keeps the JS fallback. Binary (two same-shape contiguous arrays) and
 * array-with-scalar variants.
 */

import { type DType, effectiveDType, promoteDTypes, type TypedArray } from '../dtype';
import { ArrayStorage } from '../storage';
import * as lae2Base from './bins/logaddexp2.wasm';
import * as lae2Relaxed from './bins/logaddexp2-relaxed.wasm';
import { wasmConfig } from './config';
import { useRelaxedKernels } from './detect';
import { f16InputToScratchF32, f32OutputToF16Region, resetScratchAllocator, resolveInputPtr, wasmMalloc } from './runtime';

const BASE_THRESHOLD = 32;

type BinaryFn = (aPtr: number, bPtr: number, outPtr: number, N: number) => void;
type ScalarFn = (aPtr: number, outPtr: number, N: number, scalar: number) => void;

let _bins: typeof lae2Base | null = null;
function bins(): typeof lae2Base {
  _bins ??= useRelaxedKernels() ? lae2Relaxed : lae2Base;
  return _bins;
}

const binaryKernels: Partial<Record<DType, BinaryFn>> = {
  float64: (...a) => bins().logaddexp2_f64(...a),
  float32: (...a) => bins().logaddexp2_f32(...a),
  float16: (...a) => bins().logaddexp2_f32(...a),
};
const scalarKernels: Partial<Record<DType, ScalarFn>> = {
  float64: (...a) => bins().logaddexp2_scalar_f64(...a),
  float32: (...a) => bins().logaddexp2_scalar_f32(...a),
  float16: (...a) => bins().logaddexp2_scalar_f32(...a),
};

const ctorMap: Partial<Record<DType, new (length: number) => TypedArray>> = {
  float64: Float64Array,
  float32: Float32Array,
  float16: Float32Array,
};

/** WASM logaddexp2 of two same-shape contiguous float arrays. */
export function wasmLogaddexp2(a: ArrayStorage, b: ArrayStorage): ArrayStorage | null {
  if (!a.isCContiguous || !b.isCContiguous) return null;
  if (a.dtype !== b.dtype) return null;
  if (a.size !== b.size) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = effectiveDType(promoteDTypes(a.dtype, b.dtype));
  const kernel = binaryKernels[dtype];
  if (!kernel) return null; // int/complex → JS fallback

  const isF16 = dtype === 'float16';
  const bpe = isF16 ? 4 : (ctorMap[dtype] as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const outRegion = wasmMalloc(size * bpe);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();
  let aPtr: number;
  let bPtr: number;
  if (isF16) {
    aPtr = f16InputToScratchF32(a, size);
    bPtr = f16InputToScratchF32(b, size);
  } else {
    aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, bpe);
    bPtr = resolveInputPtr(b.data, b.isWasmBacked, b.wasmPtr, b.offset, size, bpe);
  }
  kernel(aPtr, bPtr, outRegion.ptr, size);

  if (isF16) {
    const f16Region = f32OutputToF16Region(outRegion, size);
    outRegion.release();
    if (!f16Region) return null;
    return ArrayStorage.fromWasmRegion(Array.from(a.shape), dtype, f16Region, size, Float16Array as unknown as new (b: ArrayBuffer, o: number, l: number) => TypedArray);
  }
  return ArrayStorage.fromWasmRegion(Array.from(a.shape), dtype, outRegion, size, ctorMap[dtype]! as unknown as new (b: ArrayBuffer, o: number, l: number) => TypedArray);
}

/** WASM logaddexp2 with a scalar second operand. */
export function wasmLogaddexp2Scalar(a: ArrayStorage, scalar: number): ArrayStorage | null {
  if (!a.isCContiguous) return null;
  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = effectiveDType(a.dtype);
  const kernel = scalarKernels[dtype];
  if (!kernel) return null;

  const isF16 = dtype === 'float16';
  const bpe = isF16 ? 4 : (ctorMap[dtype] as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const outRegion = wasmMalloc(size * bpe);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();
  const aPtr = isF16
    ? f16InputToScratchF32(a, size)
    : resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, bpe);
  kernel(aPtr, outRegion.ptr, size, scalar);

  if (isF16) {
    const f16Region = f32OutputToF16Region(outRegion, size);
    outRegion.release();
    if (!f16Region) return null;
    return ArrayStorage.fromWasmRegion(Array.from(a.shape), dtype, f16Region, size, Float16Array as unknown as new (b: ArrayBuffer, o: number, l: number) => TypedArray);
  }
  return ArrayStorage.fromWasmRegion(Array.from(a.shape), dtype, outRegion, size, ctorMap[dtype]! as unknown as new (b: ArrayBuffer, o: number, l: number) => TypedArray);
}
