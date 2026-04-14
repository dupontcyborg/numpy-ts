/**
 * WASM-accelerated inner product kernel.
 *
 * Pure compute backend — takes ArrayStorage inputs, returns ArrayStorage or
 * null if WASM can't handle this case (unsupported dtype, non-contiguous,
 * below size threshold). The caller (linalg.ts) handles the JS fallback.
 *
 * inner(A[M,K], B[N,K]) → C[M,N] where C[i,j] = sum_k A[i,k] * B[j,k]
 */

import * as floatBase from './bins/inner_float.wasm';
import * as floatRelaxed from './bins/inner_float-relaxed.wasm';
import { inner_i64, inner_i32, inner_i16, inner_i8 } from './bins/inner_int.wasm';
import { useRelaxedKernels } from './detect';
import {
  wasmMalloc,
  resetScratchAllocator,
  resolveInputPtr,
  scratchAlloc,
  getSharedMemory,
} from './runtime';
import { ArrayStorage } from '../storage';
import { promoteDTypes, type DType, type TypedArray } from '../dtype';
import { Complex } from '../complex';

import { wasmConfig } from './config';

let _float: typeof floatBase | null = null;
function float(): typeof floatBase {
  return (_float ??= useRelaxedKernels() ? floatRelaxed : floatBase);
}

// Minimum total elements (M*K + N*K) for WASM to be worth the copy overhead.
const BASE_THRESHOLD = 32;

type WasmInnerFn = (
  aPtr: number,
  bPtr: number,
  cPtr: number,
  M: number,
  N: number,
  K: number
) => void;

type WasmComplexInnerFn = (
  aPtr: number,
  bPtr: number,
  cPtr: number,
  M: number,
  N: number,
  K: number,
  scratchPtr: number
) => void;

// Dtype -> WASM kernel function
// Complex types use separate map with scratch parameter.
const wasmKernels: Partial<Record<DType, WasmInnerFn>> = {
  float64: (...a) => float().inner_f64(...a),
  float32: (...a) => float().inner_f32(...a),
  // float16 excluded: overflow mismatch with NumPy
  int64: inner_i64,
  uint64: inner_i64,
  int32: inner_i32,
  uint32: inner_i32,
  int16: inner_i16,
  uint16: inner_i16,
  int8: inner_i8,
  uint8: inner_i8,
};

// Complex types: deinterleave → 3 real inner products (Gauss trick) → combine.
const complexKernels: Partial<Record<DType, WasmComplexInnerFn>> = {
  complex64: (...a) => float().inner_c64(...a),
  complex128: (...a) => float().inner_c128(...a),
};

// Dtype -> TypedArray constructor
type AnyTypedArrayCtor = new (length: number) => TypedArray;
const ctorMap: Partial<Record<DType, AnyTypedArrayCtor>> = {
  float64: Float64Array,
  float32: Float32Array,
  // float16: excluded
  complex128: Float64Array, // interleaved re/im
  complex64: Float32Array,
  int64: BigInt64Array,
  uint64: BigUint64Array,
  int32: Int32Array,
  uint32: Uint32Array,
  int16: Int16Array,
  uint16: Uint16Array,
  int8: Int8Array,
  uint8: Uint8Array,
};

// Complex types store 2 floats per element
const complexFactor: Partial<Record<DType, number>> = {
  complex128: 2,
  complex64: 2,
};

/**
 * WASM-accelerated inner product. Returns null if WASM can't handle this case.
 *
 * Handles the 2D general case: inner(A[M,K], B[N,K]) → C[M,N].
 * The 1D·1D case (scalar result) is also handled when both inputs are 1D.
 * The caller should fall back to JS when null is returned.
 */
export function wasmInner(
  a: ArrayStorage,
  b: ArrayStorage
): ArrayStorage | number | Complex | null {
  // Only handle cases where both are at least 1D with matching last dim
  if (a.ndim === 0 || b.ndim === 0) return null;
  if (!a.isCContiguous || !b.isCContiguous) return null;

  const resultDtype = promoteDTypes(a.dtype, b.dtype);
  const kernel = wasmKernels[resultDtype];
  const complexKernel = complexKernels[resultDtype];
  const Ctor = ctorMap[resultDtype];
  if ((!kernel && !complexKernel) || !Ctor) return null;

  const factor = complexFactor[resultDtype] ?? 1;

  const K = a.shape[a.ndim - 1]!;
  const K2 = b.shape[b.ndim - 1]!;
  if (K !== K2) return null; // Let JS throw the error

  // For ND arrays, flatten outer dims
  const M = a.ndim === 1 ? 1 : a.shape.slice(0, -1).reduce((acc, d) => acc * d, 1);
  const N = b.ndim === 1 ? 1 : b.shape.slice(0, -1).reduce((acc, d) => acc * d, 1);

  const totalElements = M * K + N * K;
  if (totalElements < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const outElements = M * N * factor;
  const outBytes = outElements * bpe;

  // For scalar result (1D · 1D), use scratch for output
  const isScalar = a.ndim === 1 && b.ndim === 1;

  if (isScalar) {
    wasmConfig.wasmCallCount++;
    resetScratchAllocator();

    const aPtr = resolveInputPtr(
      a.data,
      a.isWasmBacked,
      a.wasmPtr,
      a.offset * factor,
      M * K * factor,
      bpe
    );
    const bPtr = resolveInputPtr(
      b.data,
      b.isWasmBacked,
      b.wasmPtr,
      b.offset * factor,
      N * K * factor,
      bpe
    );
    const outPtr = scratchAlloc(outBytes);

    if (complexKernel) {
      const scratchElements = 2 * M * K + 2 * N * K + 3 * M * N;
      const scratchBytes = scratchElements * bpe;
      const scratchPtr = scratchAlloc(scratchBytes);
      complexKernel(aPtr, bPtr, outPtr, M, N, K, scratchPtr);
    } else {
      kernel!(aPtr, bPtr, outPtr, M, N, K);
    }

    const mem = getSharedMemory();
    const outView = new (Ctor as unknown as new (
      buffer: ArrayBuffer,
      byteOffset: number,
      length: number
    ) => TypedArray)(mem.buffer, outPtr, outElements);

    if (factor === 2) {
      return new Complex(
        Number((outView as Float64Array | Float32Array)[0]!),
        Number((outView as Float64Array | Float32Array)[1]!)
      );
    }
    return (outView as Float64Array | Float32Array)[0]!;
  }

  // Array result
  const outRegion = wasmMalloc(outBytes);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  const aPtr = resolveInputPtr(
    a.data,
    a.isWasmBacked,
    a.wasmPtr,
    a.offset * factor,
    M * K * factor,
    bpe
  );
  const bPtr = resolveInputPtr(
    b.data,
    b.isWasmBacked,
    b.wasmPtr,
    b.offset * factor,
    N * K * factor,
    bpe
  );

  if (complexKernel) {
    const scratchElements = 2 * M * K + 2 * N * K + 3 * M * N;
    const scratchBytes = scratchElements * bpe;
    const scratchPtr = scratchAlloc(scratchBytes);
    complexKernel(aPtr, bPtr, outRegion.ptr, M, N, K, scratchPtr);
  } else {
    kernel!(aPtr, bPtr, outRegion.ptr, M, N, K);
  }

  // Build result shape: a.shape[:-1] + b.shape[:-1]
  const resultShape = [...a.shape.slice(0, -1), ...b.shape.slice(0, -1)];
  return ArrayStorage.fromWasmRegion(
    resultShape,
    resultDtype,
    outRegion,
    outElements,
    Ctor as unknown as new (buffer: ArrayBuffer, byteOffset: number, length: number) => TypedArray
  );
}
