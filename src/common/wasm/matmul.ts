/**
 * WASM-accelerated matmul kernel.
 *
 * Pure compute backend — takes ArrayStorage inputs, returns ArrayStorage or
 * null if WASM can't handle this case (unsupported dtype, non-contiguous,
 * below size threshold). The caller (linalg.ts) handles the JS fallback.
 *
 * Handles batched matmul by looping over batch dimensions in JS and
 * dispatching each 2D slice to WASM.
 */

import { matmul_f64, matmul_f32, matmul_c128, matmul_c64 } from './bins/matmul.wasm';
import { ensureMemory, resetAllocator, copyIn, alloc, copyOut } from './runtime';
import { ArrayStorage } from '../storage';
import { promoteDTypes, type DType } from '../dtype';

// Minimum total elements (M*K + K*N) for WASM to be worth the copy overhead.
const THRESHOLD = 256;

type WasmMatmulFn = (
  aPtr: number,
  bPtr: number,
  cPtr: number,
  M: number,
  N: number,
  K: number
) => void;

// Dtype -> WASM kernel function
const wasmKernels: Partial<Record<DType, WasmMatmulFn>> = {
  float64: matmul_f64,
  float32: matmul_f32,
  complex128: matmul_c128,
  complex64: matmul_c64,
};

// Dtype -> TypedArray constructor for the underlying data
// Complex types use float arrays (interleaved re/im pairs)
const ctorMap: Record<string, Float64ArrayConstructor | Float32ArrayConstructor> = {
  float64: Float64Array,
  float32: Float32Array,
  complex128: Float64Array,
  complex64: Float32Array,
};

// Complex types store 2 floats per element
const complexFactor: Partial<Record<DType, number>> = {
  complex128: 2,
  complex64: 2,
};

/**
 * Run a single 2D matmul via WASM.
 */
function wasmMatmul2D(
  kernel: WasmMatmulFn,
  Ctor: Float64ArrayConstructor | Float32ArrayConstructor,
  aData: Float64Array | Float32Array,
  bData: Float64Array | Float32Array,
  M: number,
  K: number,
  N: number,
  factor: number = 1
): Float64Array | Float32Array {
  const bytesPerElement = Ctor.BYTES_PER_ELEMENT;
  const aBytes = M * K * factor * bytesPerElement;
  const bBytes = K * N * factor * bytesPerElement;
  const outBytes = M * N * factor * bytesPerElement;

  ensureMemory(aBytes + bBytes + outBytes);
  resetAllocator();

  const aPtr = copyIn(aData);
  const bPtr = copyIn(bData);
  const outPtr = alloc(outBytes);

  kernel(aPtr, bPtr, outPtr, M, N, K);

  return copyOut(
    outPtr,
    M * N * factor,
    Ctor as new (
      buffer: ArrayBuffer,
      byteOffset: number,
      length: number
    ) => Float64Array | Float32Array
  );
}

/**
 * WASM-accelerated matmul. Returns null if WASM can't handle this case.
 *
 * The caller should fall back to JS when null is returned.
 */
export function wasmMatmul(a: ArrayStorage, b: ArrayStorage): ArrayStorage | null {
  // 0D inputs can't be matmul'd — let JS throw the proper error
  if (a.ndim === 0 || b.ndim === 0) return null;

  // Determine the output dtype
  const resultDtype = promoteDTypes(a.dtype, b.dtype);

  // Resolve the working dtype for WASM (int types promote to float64)
  const workDtype: DType =
    resultDtype === 'float32'
      ? 'float32'
      : resultDtype === 'float64'
        ? 'float64'
        : resultDtype.startsWith('int') || resultDtype.startsWith('uint') || resultDtype === 'bool'
          ? 'float64'
          : resultDtype;

  const kernel = wasmKernels[workDtype];
  const Ctor = ctorMap[workDtype];

  // Can't handle: no WASM kernel (bigint), or non-contiguous
  if (!kernel || !Ctor || !a.isCContiguous || !b.isCContiguous) {
    return null;
  }

  // Complex types store 2 floats per element; real types store 1
  const factor = complexFactor[workDtype] ?? 1;

  // Handle 1D promotion (same as core matmul)
  const aWas1D = a.ndim === 1;
  const bWas1D = b.ndim === 1;

  const aShape = aWas1D ? [1, a.shape[0]!] : Array.from(a.shape);
  const bShape = bWas1D ? [b.shape[0]!, 1] : Array.from(b.shape);

  const aNdim = aShape.length;
  const bNdim = bShape.length;
  const M = aShape[aNdim - 2]!;
  const K = aShape[aNdim - 1]!;
  const K2 = bShape[bNdim - 2]!;
  const N = bShape[bNdim - 1]!;

  if (K !== K2) {
    return null; // Let JS throw the proper error
  }

  const totalElements = M * K + K * N;
  if (totalElements < THRESHOLD) {
    return null; // Below threshold, JS is faster
  }

  // Get contiguous data in the working dtype
  const aData = getContiguousData(a, workDtype, factor);
  const bData = getContiguousData(b, workDtype, factor);

  // --- Pure 2D case ---
  if (aNdim === 2 && bNdim === 2) {
    const outData = wasmMatmul2D(kernel, Ctor, aData, bData, M, K, N, factor);
    let outShape: number[];
    if (aWas1D && bWas1D) outShape = [];
    else if (aWas1D) outShape = [N];
    else if (bWas1D) outShape = [M];
    else outShape = [M, N];
    return ArrayStorage.fromData(outData, outShape, workDtype);
  }

  // --- Batched ND case ---
  const aBatch = aShape.slice(0, aNdim - 2);
  const bBatch = bShape.slice(0, bNdim - 2);
  const batchShape = broadcastBatchShapes(aBatch, bBatch);
  const batchSize = batchShape.reduce((acc, d) => acc * d, 1);

  const sliceA = M * K * factor;
  const sliceB = K * N * factor;
  const sliceOut = M * N * factor;
  const resultData = new Ctor(batchSize * sliceOut);

  for (let bi = 0; bi < batchSize; bi++) {
    const batchIdx = flatToBatchMultiIndex(bi, batchShape);
    const aFlatBatch = batchMultiIndexToFlat(batchIdx, aBatch);
    const bFlatBatch = batchMultiIndexToFlat(batchIdx, bBatch);

    const aOff = aFlatBatch * sliceA;
    const bOff = bFlatBatch * sliceB;

    const aSliceData = aData.subarray(aOff, aOff + sliceA);
    const bSliceData = bData.subarray(bOff, bOff + sliceB);

    const sliceResult = wasmMatmul2D(kernel, Ctor, aSliceData, bSliceData, M, K, N, factor);

    resultData.set(sliceResult, bi * sliceOut);
  }

  const outShape = [...batchShape, M, N];
  const result = ArrayStorage.fromData(resultData, outShape, workDtype);

  if (aWas1D && bWas1D) return reshapeStorage(result, [...batchShape]);
  if (aWas1D) return reshapeStorage(result, [...batchShape, N]);
  if (bWas1D) return reshapeStorage(result, [...batchShape, M]);
  return result;
}

// --- Helpers ---

function getContiguousData(
  storage: ArrayStorage,
  targetDtype: DType,
  factor: number
): Float64Array | Float32Array {
  const data = storage.data;
  const offset = storage.offset;
  const size = storage.size;
  const rawLength = size * factor;

  if (storage.dtype === targetDtype && offset === 0) {
    return data.subarray(0, rawLength) as Float64Array | Float32Array;
  }
  if (storage.dtype === targetDtype) {
    const rawOffset = offset * factor;
    return data.subarray(rawOffset, rawOffset + rawLength) as Float64Array | Float32Array;
  }

  // Type conversion needed (only for real types)
  const Ctor = targetDtype === 'float32' ? Float32Array : Float64Array;
  const result = new Ctor(rawLength);
  for (let i = 0; i < size; i++) {
    result[i] = Number(storage.iget(i));
  }
  return result;
}

function reshapeStorage(storage: ArrayStorage, newShape: number[]): ArrayStorage {
  return ArrayStorage.fromData(storage.data, newShape, storage.dtype);
}

function broadcastBatchShapes(a: number[], b: number[]): number[] {
  const maxLen = Math.max(a.length, b.length);
  const result: number[] = [];
  for (let i = 0; i < maxLen; i++) {
    const ai = i < maxLen - a.length ? 1 : a[i - (maxLen - a.length)]!;
    const bi = i < maxLen - b.length ? 1 : b[i - (maxLen - b.length)]!;
    if (ai !== bi && ai !== 1 && bi !== 1) {
      throw new Error(`matmul: batch shapes not broadcastable: [${a}] vs [${b}]`);
    }
    result.push(Math.max(ai, bi));
  }
  return result;
}

function flatToBatchMultiIndex(flat: number, shape: number[]): number[] {
  const idx: number[] = new Array(shape.length);
  let remaining = flat;
  for (let i = shape.length - 1; i >= 0; i--) {
    idx[i] = remaining % shape[i]!;
    remaining = Math.floor(remaining / shape[i]!);
  }
  return idx;
}

function batchMultiIndexToFlat(idx: number[], batchShape: number[]): number {
  const padded = idx.length - batchShape.length;
  let flat = 0;
  let stride = 1;
  for (let i = batchShape.length - 1; i >= 0; i--) {
    const dim = batchShape[i]!;
    const ii = idx[i + padded]!;
    flat += (dim === 1 ? 0 : ii) * stride;
    stride *= dim;
  }
  return flat;
}
