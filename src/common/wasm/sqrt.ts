/**
 * WASM-accelerated element-wise square root.
 *
 * Unary: out[i] = sqrt(a[i])
 * Returns null if WASM can't handle this case.
 * Float types output same type; integer types use native WASM kernels
 * that convert to f64 internally (no JS conversion loop needed).
 */

import { sqrt_f64, sqrt_f32, sqrt_i64, sqrt_i32, sqrt_i16, sqrt_i8 } from './bins/sqrt.wasm';
import { ensureMemory, resetAllocator, copyIn, alloc, copyOut } from './runtime';
import { ArrayStorage } from '../storage';
import { isComplexDType, type DType, type TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 64;

type UnaryFn = (aPtr: number, outPtr: number, N: number) => void;

// Float kernels: input & output same type
const floatKernels: Partial<Record<DType, UnaryFn>> = {
  float64: sqrt_f64,
  float32: sqrt_f32,
};

// Integer kernels: input is native int, output is f64
const intKernels: Partial<Record<DType, UnaryFn>> = {
  int64: sqrt_i64,
  uint64: sqrt_i64,
  int32: sqrt_i32,
  uint32: sqrt_i32,
  int16: sqrt_i16,
  uint16: sqrt_i16,
  int8: sqrt_i8,
  uint8: sqrt_i8,
};

type AnyTypedArrayCtor = new (length: number) => TypedArray;
const inputCtorMap: Partial<Record<DType, AnyTypedArrayCtor>> = {
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
 * WASM-accelerated element-wise square root.
 * Returns null if WASM can't handle (complex, non-contiguous, too small).
 */
export function wasmSqrt(a: ArrayStorage): ArrayStorage | null {
  if (!a.isCContiguous) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = a.dtype;
  if (isComplexDType(dtype)) return null;

  // Float path: input & output same type
  const floatKernel = floatKernels[dtype];
  if (floatKernel) {
    const Ctor = inputCtorMap[dtype]!;
    const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;

    ensureMemory(size * bpe * 2);
    resetAllocator();

    const aData = a.data.subarray(a.offset, a.offset + size) as TypedArray;
    const aPtr = copyIn(aData);
    const outPtr = alloc(size * bpe);

    floatKernel(aPtr, outPtr, size);

    const outData = copyOut(
      outPtr,
      size,
      Ctor as unknown as new (buffer: ArrayBuffer, byteOffset: number, length: number) => TypedArray
    );
    return ArrayStorage.fromData(outData, Array.from(a.shape), dtype);
  }

  // Integer path: native int input, f64 output
  const intKernel = intKernels[dtype];
  if (intKernel) {
    const InputCtor = inputCtorMap[dtype]!;
    const inputBpe = (InputCtor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
    const outBpe = 8; // f64

    ensureMemory(size * inputBpe + size * outBpe);
    resetAllocator();

    const aData = a.data.subarray(a.offset, a.offset + size) as TypedArray;
    const aPtr = copyIn(aData);
    const outPtr = alloc(size * outBpe);

    intKernel(aPtr, outPtr, size);

    const outData = copyOut(
      outPtr,
      size,
      Float64Array as unknown as new (
        buffer: ArrayBuffer,
        byteOffset: number,
        length: number
      ) => TypedArray
    );
    return ArrayStorage.fromData(outData, Array.from(a.shape), 'float64');
  }

  return null;
}
