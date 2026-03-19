/**
 * WASM-accelerated extract and take_along_axis operations.
 */

import {
  extract_f64, extract_f32, extract_i64, extract_u64,
  extract_i32, extract_u32, extract_i16, extract_u16,
  extract_i8, extract_u8,
  take_axis0_2d_f64, take_axis0_2d_f32,
  take_axis0_2d_i64, take_axis0_2d_u64,
  take_axis0_2d_i32, take_axis0_2d_u32,
  take_axis0_2d_i16, take_axis0_2d_u16,
  take_axis0_2d_i8, take_axis0_2d_u8,
} from './bins/gather.wasm';
import { ensureMemory, resetAllocator, copyIn, alloc, copyOut } from './runtime';
import { ArrayStorage } from '../storage';
import type { DType, TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 64;

type ExtractFn = (condPtr: number, dataPtr: number, outPtr: number, N: number) => number;
type TakeFn = (dataPtr: number, indicesPtr: number, outPtr: number, rows: number, cols: number) => void;

const extractKernels: Partial<Record<DType, ExtractFn>> = {
  float64: extract_f64, float32: extract_f32,
  int64: extract_i64, uint64: extract_u64,
  int32: extract_i32, uint32: extract_u32,
  int16: extract_i16, uint16: extract_u16,
  int8: extract_i8, uint8: extract_u8,
};

const takeKernels: Partial<Record<DType, TakeFn>> = {
  float64: take_axis0_2d_f64, float32: take_axis0_2d_f32,
  int64: take_axis0_2d_i64, uint64: take_axis0_2d_u64,
  int32: take_axis0_2d_i32, uint32: take_axis0_2d_u32,
  int16: take_axis0_2d_i16, uint16: take_axis0_2d_u16,
  int8: take_axis0_2d_i8, uint8: take_axis0_2d_u8,
};

type AnyTypedArrayCtor = new (length: number) => TypedArray;
const ctorMap: Partial<Record<DType, AnyTypedArrayCtor>> = {
  float64: Float64Array, float32: Float32Array,
  int64: BigInt64Array, uint64: BigUint64Array,
  int32: Int32Array, uint32: Uint32Array,
  int16: Int16Array, uint16: Uint16Array,
  int8: Int8Array, uint8: Uint8Array,
};

/**
 * WASM-accelerated extract (conditional gather).
 * condition must be flattened int32, data must be contiguous.
 * Returns ArrayStorage or null.
 */
export function wasmExtract(
  condition: ArrayStorage,
  storage: ArrayStorage
): ArrayStorage | null {
  if (!condition.isCContiguous || !storage.isCContiguous) return null;

  const size = Math.min(condition.size, storage.size);
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = storage.dtype;
  const kernel = extractKernels[dtype];
  const Ctor = ctorMap[dtype];
  if (!kernel || !Ctor) return null;

  // Condition must be a simple numeric type (not complex/bigint for the WASM kernel)
  const condDtype = condition.dtype;
  if (condDtype !== 'int32' && condDtype !== 'float64' && condDtype !== 'int8' &&
      condDtype !== 'uint8' && condDtype !== 'int16' && condDtype !== 'uint16' &&
      condDtype !== 'float32' && condDtype !== 'uint32') return null;

  const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;

  // First: count nonzero in condition to know output size
  // Convert condition to i32 for the WASM kernel
  const condOff = condition.offset;
  const condData = condition.data;
  const condI32 = new Int32Array(size);
  for (let i = 0; i < size; i++) {
    condI32[i] = condData[condOff + i] ? 1 : 0;
  }

  const condBytes = size * 4;
  const dataBytes = size * bpe;
  const outMaxBytes = size * bpe; // worst case: all selected

  ensureMemory(condBytes + dataBytes + outMaxBytes);
  resetAllocator();

  const condPtr = copyIn(condI32);

  const dataOff = storage.offset;
  const dataSlice = storage.data.subarray(dataOff, dataOff + size) as TypedArray;
  const dataPtr = copyIn(dataSlice);

  const outPtr = alloc(outMaxBytes);

  const count = kernel(condPtr, dataPtr, outPtr, size);

  const outData = copyOut(
    outPtr,
    count,
    Ctor as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
  );

  return ArrayStorage.fromData(outData, [count], dtype);
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

  const dtype = storage.dtype;
  const kernel = takeKernels[dtype];
  const Ctor = ctorMap[dtype];
  if (!kernel || !Ctor) return null;

  const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const dataBytes = totalSize * bpe;
  const idxBytes = totalSize * 4; // i32 indices
  const outBytes = totalSize * bpe;

  ensureMemory(dataBytes + idxBytes + outBytes);
  resetAllocator();

  const dataOff = storage.offset;
  const dataSlice = storage.data.subarray(dataOff, dataOff + totalSize) as TypedArray;
  const dataPtr = copyIn(dataSlice);

  // Convert indices to i32
  const idxOff = indices.offset;
  const idxData = indices.data;
  const idxI32 = new Int32Array(totalSize);
  for (let i = 0; i < totalSize; i++) {
    idxI32[i] = Number(idxData[idxOff + i]);
  }
  const idxPtr = copyIn(idxI32);

  const outPtr = alloc(outBytes);

  kernel(dataPtr, idxPtr, outPtr, rows!, cols!);

  const outData = copyOut(
    outPtr,
    totalSize,
    Ctor as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
  );

  return ArrayStorage.fromData(outData, Array.from(indices.shape), dtype);
}
