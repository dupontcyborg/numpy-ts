/**
 * WASM-accelerated extract and take_along_axis operations.
 */

import {
  extract_f64,
  extract_f32,
  extract_i64,
  extract_u64,
  extract_i32,
  extract_u32,
  extract_i16,
  extract_u16,
  extract_i8,
  extract_u8,
  take_axis0_2d_f64,
  take_axis0_2d_f32,
  take_axis0_2d_i64,
  take_axis0_2d_u64,
  take_axis0_2d_i32,
  take_axis0_2d_u32,
  take_axis0_2d_i16,
  take_axis0_2d_u16,
  take_axis0_2d_i8,
  take_axis0_2d_u8,
  where_f64,
  where_f32,
  where_i64,
  where_u64,
  where_i32,
  where_u32,
  where_i16,
  where_u16,
  where_i8,
  where_u8,
} from './bins/gather.wasm';
import {
  wasmMalloc,
  resetScratchAllocator,
  resolveInputPtr,
  scratchCopyIn,
  f16InputToScratchF32,
  f32OutputToF16Region,
} from './runtime';
import { ArrayStorage } from '../storage';
import { effectiveDType, type DType, TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 64;

type ExtractFn = (condPtr: number, dataPtr: number, outPtr: number, N: number) => number;
type TakeFn = (
  dataPtr: number,
  indicesPtr: number,
  outPtr: number,
  rows: number,
  cols: number
) => void;

const extractKernels: Partial<Record<DType, ExtractFn>> = {
  float64: extract_f64,
  float32: extract_f32,
  int64: extract_i64,
  uint64: extract_u64,
  int32: extract_i32,
  uint32: extract_u32,
  int16: extract_i16,
  uint16: extract_u16,
  int8: extract_i8,
  uint8: extract_u8,
  float16: extract_f32,
};

const takeKernels: Partial<Record<DType, TakeFn>> = {
  float64: take_axis0_2d_f64,
  float32: take_axis0_2d_f32,
  int64: take_axis0_2d_i64,
  uint64: take_axis0_2d_u64,
  int32: take_axis0_2d_i32,
  uint32: take_axis0_2d_u32,
  int16: take_axis0_2d_i16,
  uint16: take_axis0_2d_u16,
  int8: take_axis0_2d_i8,
  uint8: take_axis0_2d_u8,
  float16: take_axis0_2d_f32,
};

type AnyTypedArrayCtor = new (length: number) => TypedArray;
const ctorMap: Partial<Record<DType, AnyTypedArrayCtor>> = {
  float64: Float64Array,
  float32: Float32Array,
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

/**
 * WASM-accelerated extract (conditional gather).
 * condition must be flattened int32, data must be contiguous.
 * Returns ArrayStorage or null.
 *
 * Note: extract output size is unknown until kernel runs, so we use
 * wasmMalloc for worst-case, then trim. For the actual result we need
 * to know the count, so we read from the persistent region.
 */
export function wasmExtract(condition: ArrayStorage, storage: ArrayStorage): ArrayStorage | null {
  if (!condition.isCContiguous || !storage.isCContiguous) return null;

  const size = Math.min(condition.size, storage.size);
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = effectiveDType(storage.dtype);
  const kernel = extractKernels[dtype];
  const Ctor = ctorMap[dtype];
  if (!kernel || !Ctor) return null;

  // Condition must be a simple numeric type (not complex/bigint for the WASM kernel)
  const condDtype = condition.dtype;
  if (
    condDtype !== 'int32' &&
    condDtype !== 'float64' &&
    condDtype !== 'int8' &&
    condDtype !== 'uint8' &&
    condDtype !== 'int16' &&
    condDtype !== 'uint16' &&
    condDtype !== 'float32' &&
    condDtype !== 'uint32'
  )
    return null;

  const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;

  // Convert condition to i32 for the WASM kernel
  const condOff = condition.offset;
  const condData = condition.data;
  const condI32 = new Int32Array(size);
  for (let i = 0; i < size; i++) {
    condI32[i] = condData[condOff + i] ? 1 : 0;
  }

  const outMaxBytes = size * bpe; // worst case: all selected

  // Use wasmMalloc for output (worst case)
  const outRegion = wasmMalloc(outMaxBytes);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  const condPtr = scratchCopyIn(condI32 as unknown as TypedArray);

  const isF16 = dtype === 'float16';
  const dataOff = storage.offset;

  let dataPtr: number;
  if (isF16) {
    dataPtr = f16InputToScratchF32(storage, size);
  } else {
    dataPtr = resolveInputPtr(
      storage.data,
      storage.isWasmBacked,
      storage.wasmPtr,
      dataOff,
      size,
      bpe
    );
  }

  const count = kernel(condPtr, dataPtr, outRegion.ptr, size);

  if (isF16) {
    const f16Region = f32OutputToF16Region(outRegion, count);
    outRegion.release();
    if (!f16Region) return null;
    return ArrayStorage.fromWasmRegion(
      [count],
      dtype,
      f16Region,
      count,
      Float16Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
    );
  }

  // The output region may be larger than needed. We create a view of just the count elements.
  // Since fromWasmRegion creates a view, the extra bytes are wasted but harmless.
  return ArrayStorage.fromWasmRegion(
    [count],
    dtype,
    outRegion,
    count,
    Ctor as unknown as new (buffer: ArrayBuffer, byteOffset: number, length: number) => TypedArray
  );
}

/**
 * WASM-accelerated take_along_axis for 2D arrays along axis 0.
 * Returns ArrayStorage or null.
 */
export function wasmTakeAlongAxis2D(
  storage: ArrayStorage,
  indices: ArrayStorage,
  axis: number
): ArrayStorage | null {
  if (axis !== 0) return null;
  if (storage.ndim !== 2 || indices.ndim !== 2) return null;
  if (!storage.isCContiguous || !indices.isCContiguous) return null;

  const [rows, cols] = storage.shape;
  const totalSize = rows! * cols!;
  if (totalSize < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = effectiveDType(storage.dtype);
  const kernel = takeKernels[dtype];
  const Ctor = ctorMap[dtype];
  if (!kernel || !Ctor) return null;

  const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const outBytes = totalSize * bpe;
  const isF16 = dtype === 'float16';

  const outRegion = wasmMalloc(outBytes);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  let dataPtr: number;
  if (isF16) {
    dataPtr = f16InputToScratchF32(storage, totalSize);
  } else {
    dataPtr = resolveInputPtr(
      storage.data,
      storage.isWasmBacked,
      storage.wasmPtr,
      storage.offset,
      totalSize,
      bpe
    );
  }

  // Convert indices to i32
  const idxOff = indices.offset;
  const idxData = indices.data;
  const idxI32 = new Int32Array(totalSize);
  for (let i = 0; i < totalSize; i++) {
    idxI32[i] = Number(idxData[idxOff + i]);
  }
  const idxPtr = scratchCopyIn(idxI32 as unknown as TypedArray);

  kernel(dataPtr, idxPtr, outRegion.ptr, rows!, cols!);

  if (isF16) {
    const f16Region = f32OutputToF16Region(outRegion, totalSize);
    outRegion.release();
    if (!f16Region) return null;
    return ArrayStorage.fromWasmRegion(
      Array.from(indices.shape),
      dtype,
      f16Region,
      totalSize,
      Float16Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
    );
  }

  return ArrayStorage.fromWasmRegion(
    Array.from(indices.shape),
    dtype,
    outRegion,
    totalSize,
    Ctor as unknown as new (buffer: ArrayBuffer, byteOffset: number, length: number) => TypedArray
  );
}

// --- WASM where: out[i] = cond[i] ? x[i] : y[i] ---

type WhereFn = (condPtr: number, xPtr: number, yPtr: number, outPtr: number, N: number) => void;

const whereKernels: Partial<Record<DType, WhereFn>> = {
  float64: where_f64,
  float32: where_f32,
  int64: where_i64,
  uint64: where_u64,
  int32: where_i32,
  uint32: where_u32,
  int16: where_i16,
  uint16: where_u16,
  int8: where_i8,
  uint8: where_u8,
  float16: where_f32,
};

/**
 * WASM-accelerated element-wise where: out[i] = cond[i] ? x[i] : y[i].
 * All three arrays must be contiguous, same shape, non-complex, same dtype for x/y.
 * Condition is converted to i32 for the WASM kernel.
 * Returns ArrayStorage or null.
 */
export function wasmWhere(
  condition: ArrayStorage,
  x: ArrayStorage,
  y: ArrayStorage
): ArrayStorage | null {
  if (!condition.isCContiguous || !x.isCContiguous || !y.isCContiguous) return null;

  const size = condition.size;
  if (size !== x.size || size !== y.size) return null;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = effectiveDType(x.dtype);
  if (dtype !== y.dtype) return null;

  const kernel = whereKernels[dtype];
  const Ctor = ctorMap[dtype];
  if (!kernel || !Ctor) return null;

  const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const dataBytes = size * bpe;
  const isF16 = dtype === 'float16';

  const outRegion = wasmMalloc(dataBytes);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  // Convert condition to i32
  const condOff = condition.offset;
  const condData = condition.data;
  const condI32 = new Int32Array(size);
  for (let i = 0; i < size; i++) {
    condI32[i] = condData[condOff + i] ? 1 : 0;
  }
  const condPtr = scratchCopyIn(condI32 as unknown as TypedArray);

  let xPtr: number;
  let yPtr: number;

  if (isF16) {
    xPtr = f16InputToScratchF32(x, size);
    yPtr = f16InputToScratchF32(y, size);
  } else {
    xPtr = resolveInputPtr(x.data, x.isWasmBacked, x.wasmPtr, x.offset, size, bpe);
    yPtr = resolveInputPtr(y.data, y.isWasmBacked, y.wasmPtr, y.offset, size, bpe);
  }

  kernel(condPtr, xPtr, yPtr, outRegion.ptr, size);

  if (isF16) {
    const f16Region = f32OutputToF16Region(outRegion, size);
    outRegion.release();
    if (!f16Region) return null;
    return ArrayStorage.fromWasmRegion(
      Array.from(x.shape),
      dtype,
      f16Region,
      size,
      Float16Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
    );
  }

  return ArrayStorage.fromWasmRegion(
    Array.from(x.shape),
    dtype,
    outRegion,
    size,
    Ctor as unknown as new (buffer: ArrayBuffer, byteOffset: number, length: number) => TypedArray
  );
}
