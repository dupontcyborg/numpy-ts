/**
 * WASM-accelerated outer product.
 *
 * Computes C[i,j] = a[i] * b[j] for a[M] and b[N].
 * Returns null if WASM can't handle this case.
 */

import {
  outer_f64,
  outer_f32,
  outer_c128,
  outer_c64,
  outer_i64,
  outer_i32,
  outer_i16,
  outer_i8,
} from './bins/outer.wasm';
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

const BASE_THRESHOLD = 128; // Minimum M+N for WASM

type WasmOuterFn = (a: number, b: number, c: number, M: number, N: number) => void;

const wasmKernels: Partial<Record<DType, WasmOuterFn>> = {
  float64: outer_f64,
  float32: outer_f32,
  complex128: outer_c128,
  complex64: outer_c64,
  int64: outer_i64,
  uint64: outer_i64,
  int32: outer_i32,
  uint32: outer_i32,
  int16: outer_i16,
  uint16: outer_i16,
  int8: outer_i8,
  uint8: outer_i8,
  float16: outer_f32,
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
 * WASM-accelerated outer product: a[M] ⊗ b[N] → C[M,N].
 * Both inputs are flattened to 1D first by the caller.
 */
export function wasmOuter(a: ArrayStorage, b: ArrayStorage): ArrayStorage | null {
  if (!a.isCContiguous || !b.isCContiguous) return null;

  const M = a.size;
  const N = b.size;
  if (M + N < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const resultDtype = effectiveDType(promoteDTypes(a.dtype, b.dtype));
  const kernel = wasmKernels[resultDtype];
  const Ctor = ctorMap[resultDtype];
  if (!kernel || !Ctor) return null;

  const factor = complexFactor[resultDtype] ?? 1;
  const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const totalElements = M * N * factor;
  const outBytes = totalElements * bpe;
  const isF16 = resultDtype === 'float16';

  const outRegion = wasmMalloc(outBytes);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  if (isF16) {
    const aPtr = f16InputToScratchF32(a, M);
    const bPtr = f16InputToScratchF32(b, N);
    kernel(aPtr, bPtr, outRegion.ptr, M, N);
    const f16Region = f32OutputToF16Region(outRegion, totalElements);
    outRegion.release();
    if (!f16Region) return null;
    return ArrayStorage.fromWasmRegion(
      [M, N],
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
    M * factor,
    bpe
  );
  const bPtr = resolveInputPtr(
    b.data,
    b.isWasmBacked,
    b.wasmPtr,
    b.offset * factor,
    N * factor,
    bpe
  );

  kernel(aPtr, bPtr, outRegion.ptr, M, N);

  return ArrayStorage.fromWasmRegion(
    [M, N],
    resultDtype,
    outRegion,
    totalElements,
    Ctor as unknown as new (buffer: ArrayBuffer, byteOffset: number, length: number) => TypedArray
  );
}
