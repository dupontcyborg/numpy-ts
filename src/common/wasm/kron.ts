/**
 * WASM-accelerated Kronecker product.
 *
 * Computes C = A ⊗ B for A[am×an] and B[bm×bn].
 * Returns null if WASM can't handle this case.
 */

import {
  kron_f64,
  kron_f32,
  kron_c128,
  kron_c64,
  kron_i64,
  kron_i32,
  kron_i16,
  kron_i8,
} from './bins/kron.wasm';
import {
  wasmMalloc,
  resetScratchAllocator,
  resolveInputPtr,
  f16InputToScratchF32,
  f32OutputToF16Region,
} from './runtime';
import { ArrayStorage } from '../storage';
import { effectiveDType, promoteDTypes, type DType, type TypedArray } from '../dtype';

import { wasmConfig } from './config';

const BASE_THRESHOLD = 64; // Minimum total output elements for WASM

type WasmKronFn = (
  a: number,
  b: number,
  out: number,
  am: number,
  an: number,
  bm: number,
  bn: number
) => void;

const wasmKernels: Partial<Record<DType, WasmKronFn>> = {
  float64: kron_f64,
  float32: kron_f32,
  complex128: kron_c128,
  complex64: kron_c64,
  int64: kron_i64,
  uint64: kron_i64,
  int32: kron_i32,
  uint32: kron_i32,
  int16: kron_i16,
  uint16: kron_i16,
  int8: kron_i8,
  uint8: kron_i8,
  float16: kron_f32,
};

type AnyTypedArrayCtor = new (length: number) => TypedArray;
const ctorMap: Partial<Record<DType, AnyTypedArrayCtor>> = {
  float64: Float64Array,
  float32: Float32Array,
  complex128: Float64Array,
  complex64: Float32Array,
  int64: BigInt64Array,
  uint64: BigUint64Array,
  int32: Int32Array,
  uint32: Uint32Array,
  int16: Int16Array,
  uint16: Uint16Array,
  int8: Int8Array,
  uint8: Uint8Array,
  float16: Float32Array,
};

const complexFactor: Partial<Record<DType, number>> = {
  complex128: 2,
  complex64: 2,
};

/**
 * WASM-accelerated Kronecker product: A[am×an] ⊗ B[bm×bn] → C[(am*bm)×(an*bn)].
 * Both inputs must be 2D and contiguous.
 */
export function wasmKron(a: ArrayStorage, b: ArrayStorage): ArrayStorage | null {
  if (a.ndim !== 2 || b.ndim !== 2) return null;
  if (!a.isCContiguous || !b.isCContiguous) return null;

  const am = a.shape[0]!;
  const an = a.shape[1]!;
  const bm = b.shape[0]!;
  const bn = b.shape[1]!;

  const outRows = am * bm;
  const outCols = an * bn;
  if (outRows * outCols < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const resultDtype = effectiveDType(promoteDTypes(a.dtype, b.dtype));
  const kernel = wasmKernels[resultDtype];
  const Ctor = ctorMap[resultDtype];
  if (!kernel || !Ctor) return null;

  const factor = complexFactor[resultDtype] ?? 1;
  const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const totalElements = outRows * outCols * factor;
  const outBytes = totalElements * bpe;
  const isF16 = resultDtype === 'float16';

  const outRegion = wasmMalloc(outBytes);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  if (isF16) {
    const aPtr = f16InputToScratchF32(a, am * an);
    const bPtr = f16InputToScratchF32(b, bm * bn);
    kernel(aPtr, bPtr, outRegion.ptr, am, an, bm, bn);
    const f16Region = f32OutputToF16Region(outRegion, totalElements);
    outRegion.release();
    if (!f16Region) return null;
    return ArrayStorage.fromWasmRegion(
      [outRows, outCols],
      resultDtype,
      f16Region,
      totalElements,
      Float16Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
    );
  }

  const aPtr = resolveInputPtr(
    a.data,
    a.isWasmBacked,
    a.wasmPtr,
    a.offset * factor,
    am * an * factor,
    bpe
  );
  const bPtr = resolveInputPtr(
    b.data,
    b.isWasmBacked,
    b.wasmPtr,
    b.offset * factor,
    bm * bn * factor,
    bpe
  );

  kernel(aPtr, bPtr, outRegion.ptr, am, an, bm, bn);

  return ArrayStorage.fromWasmRegion(
    [outRows, outCols],
    resultDtype,
    outRegion,
    totalElements,
    Ctor as unknown as new (buffer: ArrayBuffer, byteOffset: number, length: number) => TypedArray
  );
}
