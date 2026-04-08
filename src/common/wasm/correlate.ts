/**
 * WASM-accelerated 1D cross-correlation (full mode).
 *
 * Computes full cross-correlation, returns float64 or float32 result.
 * The JS ops layer handles mode slicing (same/valid) and complex types.
 * Returns null if WASM can't handle this case.
 */

import {
  correlate_f64,
  correlate_f32,
  correlate_i32,
  correlate_u32,
  correlate_i16,
  correlate_u16,
  correlate_i8,
  correlate_u8,
} from './bins/correlate.wasm';
import { wasmMalloc, resetScratchAllocator, resolveInputPtr } from './runtime';
import { ArrayStorage } from '../storage';
import {
  effectiveDType,
  isComplexDType,
  isBigIntDType,
  getTypedArrayConstructor,
  promoteDTypes,
  type DType,
  TypedArray,
} from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 32;

type CorrelateFn = (
  aPtr: number,
  aLen: number,
  vPtr: number,
  vLen: number,
  outPtr: number,
  outLen: number
) => void;

const kernels: Partial<Record<DType, CorrelateFn>> = {
  float64: correlate_f64,
  float32: correlate_f32,
  int32: correlate_i32,
  uint32: correlate_u32,
  int16: correlate_i16,
  uint16: correlate_u16,
  int8: correlate_i8,
  uint8: correlate_u8,
};

const bpeMap: Partial<Record<DType, number>> = {
  float64: 8,
  float32: 4,
  int32: 4,
  uint32: 4,
  int16: 2,
  uint16: 2,
  int8: 1,
  uint8: 1,
};

/**
 * WASM-accelerated 1D cross-correlation (full mode).
 * Returns null if WASM can't handle (complex, non-contiguous, too small).
 */
export function wasmCorrelate(a: ArrayStorage, v: ArrayStorage): ArrayStorage | null {
  if (!a.isCContiguous || !v.isCContiguous) return null;

  const aLen = a.size;
  const vLen = v.size;
  const outLen = aLen + vLen - 1;

  if (outLen < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const aDtype = effectiveDType(a.dtype);
  const vDtype = effectiveDType(v.dtype);

  // Bail on complex, BigInt, bool, float16
  if (isComplexDType(aDtype) || isComplexDType(vDtype)) return null;
  if (isBigIntDType(aDtype) || isBigIntDType(vDtype)) return null;
  if (aDtype === 'bool' || vDtype === 'bool') return null;
  if (aDtype === 'float16' || vDtype === 'float16') return null;

  // Promote dtypes (e.g. int8+int16 → int16, float32+int32 → float64)
  const dtype = promoteDTypes(aDtype, vDtype);
  const kernel = kernels[dtype];
  const bpe = bpeMap[dtype];
  if (!kernel || !bpe) return null;

  const Ctor = getTypedArrayConstructor(dtype)!;
  const outBytes = outLen * bpe;

  const outRegion = wasmMalloc(outBytes);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  // Resolve or convert input pointers
  const aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, aLen, bpe);
  const vPtr = resolveInputPtr(v.data, v.isWasmBacked, v.wasmPtr, v.offset, vLen, bpe);

  kernel(aPtr, aLen, vPtr, vLen, outRegion.ptr, outLen);

  return ArrayStorage.fromWasmRegion(
    [outLen],
    dtype,
    outRegion,
    outLen,
    Ctor as unknown as new (buffer: ArrayBuffer, byteOffset: number, length: number) => TypedArray
  );
}
