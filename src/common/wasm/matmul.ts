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

import * as floatBase from './bins/matmul_float.wasm';
import * as floatRelaxed from './bins/matmul_float-relaxed.wasm';
import { matmul_i64, matmul_i32, matmul_i16, matmul_i8 } from './bins/matmul_int.wasm';
import { useRelaxedKernels } from './detect';
import {
  wasmMalloc,
  resetScratchAllocator,
  resolveTypedArrayPtr,
  scratchAlloc,
  f32OutputToF16Region,
} from './runtime';
import { ArrayStorage } from '../storage';
import {
  effectiveDType,
  isComplexDType,
  promoteDTypes,
  type DType,
  type TypedArray,
} from '../dtype';
import { wasmConfig } from './config';

// Resolve float kernel module once — relaxed if supported, baseline otherwise.
// Safe: .wasm.ts modules use lazy init(), importing doesn't instantiate.
let _float: typeof floatBase | null = null;
function float(): typeof floatBase {
  return (_float ??= useRelaxedKernels() ? floatRelaxed : floatBase);
}

// Minimum total elements (M*K + K*N) for WASM to be worth the copy overhead.
const BASE_THRESHOLD = 256;

type WasmMatmulFn = (
  aPtr: number,
  bPtr: number,
  cPtr: number,
  M: number,
  N: number,
  K: number
) => void;

type WasmComplexMatmulFn = (
  aPtr: number,
  bPtr: number,
  cPtr: number,
  M: number,
  N: number,
  K: number,
  scratchPtr: number
) => void;

// Dtype -> WASM kernel function
// Signed and unsigned integer types share the same kernel per bit-width
// (wrapping add/mul produce identical bits regardless of sign interpretation).
// Complex types use Gauss-trick kernels (3 real matmuls) — see complexKernels.
const wasmKernels: Partial<Record<DType, WasmMatmulFn>> = {
  float64: (...a) => float().matmul_f64(...a),
  float32: (...a) => float().matmul_f32(...a),
  int64: matmul_i64,
  uint64: matmul_i64,
  int32: matmul_i32,
  uint32: matmul_i32,
  int16: matmul_i16,
  uint16: matmul_i16,
  int8: matmul_i8,
  uint8: matmul_i8,
  float16: (...a) => float().matmul_f32(...a),
};

// Complex types: deinterleave → 3 real matmuls (Gauss trick) → combine + reinterleave.
// Takes an extra scratch pointer. ~30-40% faster than the old interleaved kernels.
const complexKernels: Partial<Record<DType, WasmComplexMatmulFn>> = {
  complex64: (...a) => float().matmul_c64(...a),
  complex128: (...a) => float().matmul_c128(...a),
};

// Dtype -> TypedArray constructor for the underlying data
// Complex types use float arrays (interleaved re/im pairs)
// Integer types: signed constructor for both signed/unsigned (bit-identical in WASM)
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

// Complex types store 2 floats per element
const complexFactor: Partial<Record<DType, number>> = {
  complex128: 2,
  complex64: 2,
};

/**
 * Run a single 2D matmul via WASM (real / integer types).
 * Writes result into outPtr (must be pre-allocated in persistent or scratch space).
 */
function wasmMatmul2DInto(
  kernel: WasmMatmulFn,
  aData: TypedArray,
  bData: TypedArray,
  outPtr: number,
  M: number,
  K: number,
  N: number
): void {
  resetScratchAllocator();
  const aPtr = resolveTypedArrayPtr(aData);
  const bPtr = resolveTypedArrayPtr(bData);
  kernel(aPtr, bPtr, outPtr, M, N, K);
}

/**
 * Run a single 2D complex matmul via Gauss-trick WASM kernel.
 * Writes result into outPtr. Uses scratch for intermediates.
 */
function wasmMatmul2DComplexInto(
  kernel: WasmComplexMatmulFn,
  Ctor: AnyTypedArrayCtor,
  aData: TypedArray,
  bData: TypedArray,
  outPtr: number,
  M: number,
  K: number,
  N: number
): void {
  const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const scratchElements = 2 * M * K + 2 * K * N + 3 * M * N;
  const scratchBytes = scratchElements * bpe;

  resetScratchAllocator();
  const aPtr = resolveTypedArrayPtr(aData);
  const bPtr = resolveTypedArrayPtr(bData);
  const scratchPtr = scratchAlloc(scratchBytes);
  kernel(aPtr, bPtr, outPtr, M, N, K, scratchPtr);
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
  const resultDtype = effectiveDType(promoteDTypes(a.dtype, b.dtype));

  // Bool: no WASM kernel
  if (resultDtype === 'bool') return null;

  // Mixed real/complex: WASM complex kernel expects both inputs interleaved
  // TODO: support mixed dtypes by upcasting here
  if (isComplexDType(resultDtype) && a.dtype !== b.dtype) return null;

  const workDtype: DType = resultDtype;

  const kernel = wasmKernels[workDtype];
  const gaussKernel = complexKernels[workDtype];
  const Ctor = ctorMap[workDtype];

  // Can't handle: no WASM kernel, or non-contiguous
  if ((!kernel && !gaussKernel) || !Ctor || !a.isCContiguous || !b.isCContiguous) {
    return null;
  }

  // Complex types store 2 floats per element; real types store 1
  const factor = complexFactor[workDtype] ?? 1;
  const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;

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
  if (totalElements < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) {
    return null; // Below threshold, JS is faster
  }

  // Get contiguous data in the working dtype
  const isF16 = workDtype === 'float16';
  let aData = getContiguousData(a, workDtype, factor);
  let bData = getContiguousData(b, workDtype, factor);
  if (isF16) {
    aData = new Float32Array(aData as unknown as ArrayLike<number>) as unknown as TypedArray;
    bData = new Float32Array(bData as unknown as ArrayLike<number>) as unknown as TypedArray;
  }

  // --- Pure 2D case ---
  if (aNdim === 2 && bNdim === 2) {
    const outElements = M * N * factor;
    const outBytes = outElements * bpe;
    const outRegion = wasmMalloc(outBytes);
    if (!outRegion) return null;

    wasmConfig.wasmCallCount++;

    if (gaussKernel) {
      wasmMatmul2DComplexInto(gaussKernel, Ctor, aData, bData, outRegion.ptr, M, K, N);
    } else {
      wasmMatmul2DInto(kernel!, aData, bData, outRegion.ptr, M, K, N);
    }

    if (isF16) {
      let outShape: number[];
      if (aWas1D && bWas1D) outShape = [];
      else if (aWas1D) outShape = [N];
      else if (bWas1D) outShape = [M];
      else outShape = [M, N];
      const f16Region = f32OutputToF16Region(outRegion, outElements);
      outRegion.release();
      if (!f16Region) return null;
      return ArrayStorage.fromWasmRegion(
        outShape,
        workDtype,
        f16Region,
        outElements,
        Float16Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
      );
    }

    let outShape: number[];
    if (aWas1D && bWas1D) outShape = [];
    else if (aWas1D) outShape = [N];
    else if (bWas1D) outShape = [M];
    else outShape = [M, N];

    const result = ArrayStorage.fromWasmRegion(
      outShape.length === 0 ? [M, N] : outShape,
      workDtype,
      outRegion,
      outElements,
      Ctor as unknown as new (buffer: ArrayBuffer, byteOffset: number, length: number) => TypedArray
    );

    // For scalar/1D results, reshape
    if (aWas1D && bWas1D) return reshapeStorage(result, []);
    if (aWas1D) return reshapeStorage(result, [N]);
    if (bWas1D) return reshapeStorage(result, [M]);
    return result;
  }

  // --- Batched ND case ---
  const aBatch = aShape.slice(0, aNdim - 2);
  const bBatch = bShape.slice(0, bNdim - 2);
  const batchShape = broadcastBatchShapes(aBatch, bBatch);
  const batchSize = batchShape.reduce((acc, d) => acc * d, 1);

  const sliceA = M * K * factor;
  const sliceB = K * N * factor;
  const sliceOut = M * N * factor;
  const totalOut = batchSize * sliceOut;
  const outBytes = totalOut * bpe;

  const outRegion = wasmMalloc(outBytes);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;

  for (let bi = 0; bi < batchSize; bi++) {
    const batchIdx = flatToBatchMultiIndex(bi, batchShape);
    const aFlatBatch = batchMultiIndexToFlat(batchIdx, aBatch);
    const bFlatBatch = batchMultiIndexToFlat(batchIdx, bBatch);

    const aOff = aFlatBatch * sliceA;
    const bOff = bFlatBatch * sliceB;

    const aSliceData = aData.subarray(aOff, aOff + sliceA) as TypedArray;
    const bSliceData = bData.subarray(bOff, bOff + sliceB) as TypedArray;

    const sliceOutPtr = outRegion.ptr + bi * sliceOut * bpe;

    if (gaussKernel) {
      wasmMatmul2DComplexInto(gaussKernel, Ctor, aSliceData, bSliceData, sliceOutPtr, M, K, N);
    } else {
      wasmMatmul2DInto(kernel!, aSliceData, bSliceData, sliceOutPtr, M, K, N);
    }
  }

  const outShape = [...batchShape, M, N];

  if (isF16) {
    const f16Region = f32OutputToF16Region(outRegion, totalOut);
    outRegion.release();
    if (!f16Region) return null;
    const result = ArrayStorage.fromWasmRegion(
      outShape,
      workDtype,
      f16Region,
      totalOut,
      Float16Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
    );
    if (aWas1D && bWas1D) return reshapeStorage(result, [...batchShape]);
    if (aWas1D) return reshapeStorage(result, [...batchShape, N]);
    if (bWas1D) return reshapeStorage(result, [...batchShape, M]);
    return result;
  }

  const result = ArrayStorage.fromWasmRegion(
    outShape,
    workDtype,
    outRegion,
    totalOut,
    Ctor as unknown as new (buffer: ArrayBuffer, byteOffset: number, length: number) => TypedArray
  );

  if (aWas1D && bWas1D) return reshapeStorage(result, [...batchShape]);
  if (aWas1D) return reshapeStorage(result, [...batchShape, N]);
  if (bWas1D) return reshapeStorage(result, [...batchShape, M]);
  return result;
}

// --- Helpers ---

function getContiguousData(storage: ArrayStorage, targetDtype: DType, factor: number): TypedArray {
  const data = storage.data;
  const offset = storage.offset;
  const size = storage.size;
  const rawLength = size * factor;

  if (storage.dtype === targetDtype && offset === 0) {
    return data.subarray(0, rawLength) as TypedArray;
  }
  if (storage.dtype === targetDtype) {
    const rawOffset = offset * factor;
    return data.subarray(rawOffset, rawOffset + rawLength) as TypedArray;
  }

  // Type conversion needed (only for real types)
  const Ctor = ctorMap[targetDtype];
  if (!Ctor) throw new Error(`No TypedArray constructor for dtype ${targetDtype}`);
  const result = new Ctor(rawLength);
  for (let i = 0; i < size; i++) {
    (result as Int32Array)[i] = Number(storage.iget(i));
  }
  return result as TypedArray;
}

function reshapeStorage(storage: ArrayStorage, newShape: number[]): ArrayStorage {
  // Compute C-contiguous strides for the new shape
  const strides = new Array(newShape.length);
  let stride = 1;
  for (let i = newShape.length - 1; i >= 0; i--) {
    strides[i] = stride;
    stride *= newShape[i]!;
  }
  return ArrayStorage.fromDataShared(
    storage.data,
    newShape,
    storage.dtype,
    strides,
    0,
    storage.wasmRegion
  );
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
