/**
 * WASM-accelerated vector-matrix product.
 *
 * Computes y[j] = sum_k x[k] * A[k,j] for x[K] and A[K,N].
 * Returns null if WASM can't handle this case.
 */

import * as floatBase from './bins/vecmat_float.wasm';
import * as floatRelaxed from './bins/vecmat_float-relaxed.wasm';
import { vecmat_i64, vecmat_i32, vecmat_i16, vecmat_i8 } from './bins/vecmat_int.wasm';
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

const BASE_THRESHOLD = 32; // Minimum K*N for WASM

type WasmVecmatFn = (x: number, A: number, y: number, K: number, N: number) => void;

const wasmKernels: Partial<Record<DType, WasmVecmatFn>> = {
  float64: (...a) => float().vecmat_f64(...a),
  float32: (...a) => float().vecmat_f32(...a),
  complex128: (...a) => float().vecmat_c128(...a),
  complex64: (...a) => float().vecmat_c64(...a),
  int64: vecmat_i64,
  uint64: vecmat_i64,
  int32: vecmat_i32,
  uint32: vecmat_i32,
  int16: vecmat_i16,
  uint16: vecmat_i16,
  int8: vecmat_i8,
  uint8: vecmat_i8,
  float16: (...a) => float().vecmat_f32(...a),
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
 * WASM-accelerated vecmat: x[K] · A[K,N] → y[N].
 * x must be 1D, A must be 2D, both contiguous.
 */
export function wasmVecmat(x: ArrayStorage, A: ArrayStorage): ArrayStorage | null {
  if (x.ndim !== 1 || A.ndim !== 2) return null;
  if (!x.isCContiguous || !A.isCContiguous) return null;

  const K = A.shape[0]!;
  const N = A.shape[1]!;
  if (K !== x.shape[0]!) return null;
  if (K * N < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const resultDtype = effectiveDType(promoteDTypes(x.dtype, A.dtype));
  const kernel = wasmKernels[resultDtype];
  const Ctor = ctorMap[resultDtype];
  if (!kernel || !Ctor) return null;

  const factor = complexFactor[resultDtype] ?? 1;
  const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const totalElements = N * factor;
  const outBytes = totalElements * bpe;
  const isF16 = resultDtype === 'float16';

  const outRegion = wasmMalloc(outBytes);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  if (isF16) {
    const xPtr = f16InputToScratchF32(x, K);
    const aPtr = f16InputToScratchF32(A, K * N);
    kernel(xPtr, aPtr, outRegion.ptr, K, N);
    const f16Region = f32OutputToF16Region(outRegion, totalElements);
    outRegion.release();
    if (!f16Region) return null;
    return ArrayStorage.fromWasmRegion(
      [N],
      resultDtype,
      f16Region,
      totalElements,
      Float16Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
    );
  }

  const xPtr = resolveInputPtr(
    x.data,
    x.isWasmBacked,
    x.wasmPtr,
    x.offset * factor,
    K * factor,
    bpe
  );
  const aPtr = resolveInputPtr(
    A.data,
    A.isWasmBacked,
    A.wasmPtr,
    A.offset * factor,
    K * N * factor,
    bpe
  );

  kernel(xPtr, aPtr, outRegion.ptr, K, N);

  return ArrayStorage.fromWasmRegion(
    [N],
    resultDtype,
    outRegion,
    totalElements,
    Ctor as unknown as new (buffer: ArrayBuffer, byteOffset: number, length: number) => TypedArray
  );
}
