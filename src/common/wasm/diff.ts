/**
 * WASM-accelerated diff along last axis.
 *
 * 1D: flat diff.
 * nD: uses 2D kernel (rows x cols) for last-axis diffs in one WASM call.
 * Returns null if WASM can't handle this case.
 */

import {
  diff_f64,
  diff_f32,
  diff_i64,
  diff_i32,
  diff_i16,
  diff_i8,
  diff_2d_f64,
  diff_2d_f32,
  diff_2d_i64,
  diff_2d_i32,
  diff_2d_i16,
  diff_2d_i8,
} from './bins/diff.wasm';
import { ensureMemory, resetAllocator, copyIn, alloc, copyOut } from './runtime';
import { ArrayStorage } from '../storage';
import type { DType, TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 64;

type DiffFn = (aPtr: number, outPtr: number, N: number) => void;
type Diff2DFn = (aPtr: number, outPtr: number, rows: number, cols: number) => void;

// uint types reuse signed kernels: two's complement wrapping subtraction
// produces identical bit patterns regardless of sign interpretation.
const kernels1D: Partial<Record<DType, DiffFn>> = {
  float64: diff_f64,
  float32: diff_f32,
  int64: diff_i64, uint64: diff_i64,
  int32: diff_i32, uint32: diff_i32,
  int16: diff_i16, uint16: diff_i16,
  int8: diff_i8,  uint8: diff_i8,
};

const kernels2D: Partial<Record<DType, Diff2DFn>> = {
  float64: diff_2d_f64,
  float32: diff_2d_f32,
  int64: diff_2d_i64, uint64: diff_2d_i64,
  int32: diff_2d_i32, uint32: diff_2d_i32,
  int16: diff_2d_i16, uint16: diff_2d_i16,
  int8: diff_2d_i8,  uint8: diff_2d_i8,
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
};

/**
 * WASM-accelerated diff along last axis for C-contiguous arrays.
 * Handles 1D (flat) and nD (batched rows along last axis).
 * Returns null if WASM can't handle.
 */
export function wasmDiff(a: ArrayStorage, axis: number): ArrayStorage | null {
  if (!a.isCContiguous) return null;

  const shape = a.shape;
  const ndim = shape.length;
  const normalizedAxis = axis < 0 ? ndim + axis : axis;

  // Only handle last-axis diffs (contiguous rows)
  if (normalizedAxis !== ndim - 1) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = a.dtype;
  const Ctor = ctorMap[dtype];
  if (!Ctor) return null;

  const axisLen = shape[normalizedAxis]!;
  const outAxisLen = axisLen - 1;
  if (outAxisLen <= 0) return null;

  const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const numRows = size / axisLen;
  const outSize = numRows * outAxisLen;
  const aBytes = size * bpe;
  const outBytes = outSize * bpe;

  ensureMemory(aBytes + outBytes);
  resetAllocator();

  const aOff = a.offset;
  const aData = a.data.subarray(aOff, aOff + size) as TypedArray;
  const aPtr = copyIn(aData);
  const outPtr = alloc(outBytes);

  const resultShape = Array.from(shape);
  resultShape[normalizedAxis] = outAxisLen;

  if (ndim === 1) {
    const kernel = kernels1D[dtype];
    if (!kernel) return null;
    kernel(aPtr, outPtr, outAxisLen);
  } else {
    const kernel = kernels2D[dtype];
    if (!kernel) return null;
    kernel(aPtr, outPtr, numRows, axisLen);
  }

  const outData = copyOut(
    outPtr,
    outSize,
    Ctor as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
  );

  return ArrayStorage.fromData(outData, resultShape, dtype);
}
