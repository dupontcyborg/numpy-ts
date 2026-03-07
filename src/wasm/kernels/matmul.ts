/**
 * WASM-accelerated matmul kernel wrapper.
 *
 * Dispatches to WASM for contiguous float32/float64 matrices above
 * the size threshold. Handles batched matmul by looping over batch
 * dimensions in JS and dispatching each 2D slice to WASM.
 *
 * Falls back to JS for: small arrays, non-contiguous, unsupported dtypes,
 * 1D inputs, and complex dtypes (until complex WASM kernels are added).
 */

import { matmul as jsMatmul } from '../../core/linalg';
import { matmul_f64, matmul_f32, matmul_c128, matmul_c64 } from '../bins/matmul.wasm';
import { ensureMemory, resetAllocator, copyIn, alloc, copyOut, type DType } from '../runtime';
import { NDArrayCore } from '../../common/ndarray-core';
import { ArrayStorage } from '../../common/storage';
import { promoteDTypes } from '../../common/dtype';

// Minimum total elements (M*K + K*N) for WASM to be worth the copy overhead.
// matmul is O(M*K*N) so WASM wins at relatively small sizes.
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
 * Caller must ensure dtype is supported, inputs are contiguous, and size is above threshold.
 * aData/bData are typed arrays positioned at the correct offset for this slice.
 *
 * For complex types, `factor` is 2 (each element is 2 floats: re, im).
 * The kernel receives M, K, N in *elements* — it internally strides by 2.
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

export function matmul(a: NDArrayCore, b: NDArrayCore): NDArrayCore {
  const aStorage = a.storage;
  const bStorage = b.storage;

  // Determine the output dtype
  const resultDtype = promoteDTypes(aStorage.dtype, bStorage.dtype);

  // Resolve the working dtype for WASM (int types promote to float64)
  const workDtype: DType =
    resultDtype === 'float32'
      ? 'float32'
      : resultDtype === 'float64'
        ? 'float64'
        : // int/uint/bool promote to float64 for matmul
          resultDtype.startsWith('int') || resultDtype.startsWith('uint') || resultDtype === 'bool'
          ? 'float64'
          : resultDtype;

  const kernel = wasmKernels[workDtype];
  const Ctor = ctorMap[workDtype];

  // Fall back to JS if: no WASM kernel (bigint), or non-contiguous
  if (!kernel || !Ctor || !aStorage.isCContiguous || !bStorage.isCContiguous) {
    return jsMatmul(a, b);
  }

  // Complex types store 2 floats per element; real types store 1
  const factor = complexFactor[workDtype] ?? 1;

  // Handle 1D promotion (same as core matmul)
  const aWas1D = aStorage.ndim === 1;
  const bWas1D = bStorage.ndim === 1;

  const aShape = aWas1D ? [1, aStorage.shape[0]!] : Array.from(aStorage.shape);
  const bShape = bWas1D ? [bStorage.shape[0]!, 1] : Array.from(bStorage.shape);

  const aNdim = aShape.length;
  const bNdim = bShape.length;
  const M = aShape[aNdim - 2]!;
  const K = aShape[aNdim - 1]!;
  const K2 = bShape[bNdim - 2]!;
  const N = bShape[bNdim - 1]!;

  if (K !== K2) {
    // Let JS matmul throw the proper error
    return jsMatmul(a, b);
  }

  const totalElements = M * K + K * N;
  if (totalElements < THRESHOLD) {
    return jsMatmul(a, b);
  }

  // Get contiguous data in the working dtype
  const aData = getContiguousData(aStorage, workDtype);
  const bData = getContiguousData(bStorage, workDtype);

  // --- Pure 2D case ---
  if (aNdim === 2 && bNdim === 2) {
    const outData = wasmMatmul2D(kernel, Ctor, aData, bData, M, K, N, factor);
    let outShape: number[];
    if (aWas1D && bWas1D) outShape = [];
    else if (aWas1D) outShape = [N];
    else if (bWas1D) outShape = [M];
    else outShape = [M, N];
    return NDArrayCore.fromStorage(ArrayStorage.fromData(outData, outShape, workDtype));
  }

  // --- Batched ND case ---
  // Broadcast batch dimensions, then loop over batches calling WASM for each 2D slice
  const aBatch = aShape.slice(0, aNdim - 2);
  const bBatch = bShape.slice(0, bNdim - 2);
  const batchShape = broadcastBatchShapes(aBatch, bBatch);
  const batchSize = batchShape.reduce((acc, d) => acc * d, 1);

  // For complex types, each element is `factor` floats
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

  if (aWas1D && bWas1D) return NDArrayCore.fromStorage(reshapeStorage(result, [...batchShape]));
  if (aWas1D) return NDArrayCore.fromStorage(reshapeStorage(result, [...batchShape, N]));
  if (bWas1D) return NDArrayCore.fromStorage(reshapeStorage(result, [...batchShape, M]));
  return NDArrayCore.fromStorage(result);
}

// --- Helpers ---

/**
 * Get contiguous data in the target dtype, converting if necessary.
 * For complex types, returns the raw interleaved float data (2 floats per element).
 */
function getContiguousData(storage: ArrayStorage, targetDtype: DType): Float64Array | Float32Array {
  const data = storage.data;
  const offset = storage.offset;
  const size = storage.size;
  // For complex types, underlying data has 2 floats per element
  const factor = complexFactor[targetDtype] ?? 1;
  const rawLength = size * factor;

  if (storage.dtype === targetDtype && offset === 0) {
    return data.subarray(0, rawLength) as Float64Array | Float32Array;
  }
  if (storage.dtype === targetDtype) {
    // Complex offset is in element units; raw data offset is element * factor
    const rawOffset = offset * factor;
    return data.subarray(rawOffset, rawOffset + rawLength) as Float64Array | Float32Array;
  }

  // Type conversion needed (only for real types — complex-to-real conversion not supported)
  const Ctor = targetDtype === 'float32' ? Float32Array : Float64Array;
  const result = new Ctor(rawLength);
  for (let i = 0; i < size; i++) {
    result[i] = Number(storage.iget(i));
  }
  return result;
}

/** Reshape storage (creates a view with new shape, no data copy) */
function reshapeStorage(storage: ArrayStorage, newShape: number[]): ArrayStorage {
  return ArrayStorage.fromData(storage.data, newShape, storage.dtype);
}

/** Broadcast two batch shape arrays to a common shape */
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

/** Convert flat batch index to multi-index */
function flatToBatchMultiIndex(flat: number, shape: number[]): number[] {
  const idx: number[] = new Array(shape.length);
  let remaining = flat;
  for (let i = shape.length - 1; i >= 0; i--) {
    idx[i] = remaining % shape[i]!;
    remaining = Math.floor(remaining / shape[i]!);
  }
  return idx;
}

/** Convert multi-index to flat batch index (with broadcasting: clamp to dim size) */
function batchMultiIndexToFlat(idx: number[], batchShape: number[]): number {
  const padded = idx.length - batchShape.length;
  let flat = 0;
  let stride = 1;
  for (let i = batchShape.length - 1; i >= 0; i--) {
    const dim = batchShape[i]!;
    const ii = idx[i + padded]!;
    // Clamp for broadcasting: if dim is 1, always use index 0
    flat += (dim === 1 ? 0 : ii) * stride;
    stride *= dim;
  }
  return flat;
}
