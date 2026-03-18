/**
 * WASM-accelerated element-wise right shift.
 *
 * Binary: out[i] = a[i] >> b[i]  (same-shape contiguous arrays)
 * Scalar: out[i] = a[i] >> scalar
 * Returns null if WASM can't handle this case.
 */

import {
  right_shift_i64,
  right_shift_i32,
  right_shift_i16,
  right_shift_i8,
  right_shift_u64,
  right_shift_u32,
  right_shift_u16,
  right_shift_u8,
  right_shift_scalar_i64,
  right_shift_scalar_i32,
  right_shift_scalar_i16,
  right_shift_scalar_i8,
  right_shift_scalar_u64,
  right_shift_scalar_u32,
  right_shift_scalar_u16,
  right_shift_scalar_u8,
} from './bins/right_shift.wasm';
import { ensureMemory, resetAllocator, copyIn, alloc, copyOut } from './runtime';
import { ArrayStorage } from '../storage';
import { promoteDTypes, type DType, type TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 64;

type BinaryFn = (aPtr: number, bPtr: number, outPtr: number, N: number) => void;
type ScalarFn = (aPtr: number, outPtr: number, N: number, scalar: number) => void;

const binaryKernels: Partial<Record<DType, BinaryFn>> = {
  int64: right_shift_i64,
  uint64: right_shift_u64,
  int32: right_shift_i32,
  uint32: right_shift_u32,
  int16: right_shift_i16,
  uint16: right_shift_u16,
  int8: right_shift_i8,
  uint8: right_shift_u8,
};

const scalarKernels: Partial<Record<DType, ScalarFn>> = {
  int64: right_shift_scalar_i64,
  uint64: right_shift_scalar_u64,
  int32: right_shift_scalar_i32,
  uint32: right_shift_scalar_u32,
  int16: right_shift_scalar_i16,
  uint16: right_shift_scalar_u16,
  int8: right_shift_scalar_i8,
  uint8: right_shift_scalar_u8,
};

type AnyTypedArrayCtor = new (length: number) => TypedArray;
const ctorMap: Partial<Record<DType, AnyTypedArrayCtor>> = {
  int64: BigInt64Array,
  uint64: BigUint64Array,
  int32: Int32Array,
  uint32: Uint32Array,
  int16: Int16Array,
  uint16: Uint16Array,
  int8: Int8Array,
  uint8: Uint8Array,
};

export function wasmRightShift(a: ArrayStorage, b: ArrayStorage): ArrayStorage | null {
  if (!a.isCContiguous || !b.isCContiguous) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = promoteDTypes(a.dtype, b.dtype);
  const kernel = binaryKernels[dtype];
  const Ctor = ctorMap[dtype];
  if (!kernel || !Ctor) return null;

  const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const aBytes = size * bpe;
  const bBytes = size * bpe;
  const outBytes = size * bpe;

  ensureMemory(aBytes + bBytes + outBytes);
  resetAllocator();

  const aOff = a.offset;
  const bOff = b.offset;
  const aData = a.data.subarray(aOff, aOff + size) as TypedArray;
  const bData = b.data.subarray(bOff, bOff + size) as TypedArray;

  const aPtr = copyIn(aData);
  const bPtr = copyIn(bData);
  const outPtr = alloc(outBytes);

  kernel(aPtr, bPtr, outPtr, size);

  const outData = copyOut(
    outPtr,
    size,
    Ctor as unknown as new (buffer: ArrayBuffer, byteOffset: number, length: number) => TypedArray
  );

  return ArrayStorage.fromData(outData, Array.from(a.shape), dtype);
}

export function wasmRightShiftScalar(a: ArrayStorage, scalar: number): ArrayStorage | null {
  if (!a.isCContiguous) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = a.dtype;
  const kernel = scalarKernels[dtype];
  const Ctor = ctorMap[dtype];
  if (!kernel || !Ctor) return null;

  const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const aBytes = size * bpe;
  const outBytes = size * bpe;

  ensureMemory(aBytes + outBytes);
  resetAllocator();

  const aOff = a.offset;
  const aData = a.data.subarray(aOff, aOff + size) as TypedArray;

  const aPtr = copyIn(aData);
  const outPtr = alloc(outBytes);

  kernel(aPtr, outPtr, size, scalar);

  const outData = copyOut(
    outPtr,
    size,
    Ctor as unknown as new (buffer: ArrayBuffer, byteOffset: number, length: number) => TypedArray
  );

  return ArrayStorage.fromData(outData, Array.from(a.shape), dtype);
}
