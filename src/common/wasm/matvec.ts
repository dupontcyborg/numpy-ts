/**
 * WASM-accelerated matrix-vector product.
 *
 * Computes y[i] = sum_k A[i,k] * x[k] for A[M,K] and x[K].
 * Returns null if WASM can't handle this case.
 */

import * as floatBase from './bins/matvec_float.wasm';
import * as floatRelaxed from './bins/matvec_float-relaxed.wasm';
import { matvec_i64, matvec_i32, matvec_i16, matvec_i8 } from './bins/matvec_int.wasm';
import { useRelaxedKernels } from './detect';
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

let _float: typeof floatBase | null = null;
function float(): typeof floatBase {
  return (_float ??= useRelaxedKernels() ? floatRelaxed : floatBase);
}

const BASE_THRESHOLD = 32; // Minimum M*K for WASM

type WasmMatvecFn = (A: number, x: number, y: number, M: number, K: number) => void;

const wasmKernels: Partial<Record<DType, WasmMatvecFn>> = {
  float64: (...a) => float().matvec_f64(...a),
  float32: (...a) => float().matvec_f32(...a),
  complex128: (...a) => float().matvec_c128(...a),
  complex64: (...a) => float().matvec_c64(...a),
  int64: matvec_i64,
  uint64: matvec_i64,
  int32: matvec_i32,
  uint32: matvec_i32,
  int16: matvec_i16,
  uint16: matvec_i16,
  int8: matvec_i8,
  uint8: matvec_i8,
  float16: (...a) => float().matvec_f32(...a),
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
 * WASM-accelerated matvec: A[M,K] · x[K] → y[M].
 * A must be 2D, x must be 1D, both contiguous.
 */
export function wasmMatvec(A: ArrayStorage, x: ArrayStorage): ArrayStorage | null {
  if (A.ndim !== 2 || x.ndim !== 1) return null;
  if (!A.isCContiguous || !x.isCContiguous) return null;

  const M = A.shape[0]!;
  const K = A.shape[1]!;
  if (K !== x.shape[0]!) return null;
  if (M * K < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const resultDtype = effectiveDType(promoteDTypes(A.dtype, x.dtype));
  const kernel = wasmKernels[resultDtype];
  const Ctor = ctorMap[resultDtype];
  if (!kernel || !Ctor) return null;

  const factor = complexFactor[resultDtype] ?? 1;
  const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const totalElements = M * factor;
  const outBytes = totalElements * bpe;
  const isF16 = resultDtype === 'float16';

  const outRegion = wasmMalloc(outBytes);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  if (isF16) {
    const aPtr = f16InputToScratchF32(A, M * K);
    const xPtr = f16InputToScratchF32(x, K);
    kernel(aPtr, xPtr, outRegion.ptr, M, K);
    const f16Region = f32OutputToF16Region(outRegion, totalElements);
    outRegion.release();
    if (!f16Region) return null;
    return ArrayStorage.fromWasmRegion(
      [M],
      resultDtype,
      f16Region,
      totalElements,
      Float16Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
    );
  }

  const aPtr = resolveInputPtr(
    A.data,
    A.isWasmBacked,
    A.wasmPtr,
    A.offset * factor,
    M * K * factor,
    bpe
  );
  const xPtr = resolveInputPtr(
    x.data,
    x.isWasmBacked,
    x.wasmPtr,
    x.offset * factor,
    K * factor,
    bpe
  );

  kernel(aPtr, xPtr, outRegion.ptr, M, K);

  return ArrayStorage.fromWasmRegion(
    [M],
    resultDtype,
    outRegion,
    totalElements,
    Ctor as unknown as new (buffer: ArrayBuffer, byteOffset: number, length: number) => TypedArray
  );
}
