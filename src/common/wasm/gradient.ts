/**
 * WASM-accelerated 1D gradient using central differences.
 *
 * Only handles 1D arrays with uniform spacing.
 * Returns null if WASM can't handle this case.
 */

import {
  gradient_f64,
  gradient_f32,
  gradient_i64,
  gradient_i32,
  gradient_i16,
  gradient_i8,
  gradient_u64,
  gradient_u32,
  gradient_u16,
  gradient_u8,
} from './bins/gradient.wasm';
import {
  wasmMalloc,
  resetScratchAllocator,
  resolveInputPtr,
  f16InputToScratchF32,
  f32OutputToF16Region,
} from './runtime';
import { ArrayStorage } from '../storage';
import { effectiveDType, hasFloat16, type DType, TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 64;

type GradFn = (aPtr: number, outPtr: number, N: number, h: number) => void;

const kernels: Partial<Record<DType, GradFn>> = {
  float64: gradient_f64,
  float32: gradient_f32,
  int64: gradient_i64,
  uint64: gradient_u64,
  int32: gradient_i32,
  uint32: gradient_u32,
  int16: gradient_i16,
  uint16: gradient_u16,
  int8: gradient_i8,
  uint8: gradient_u8,
  float16: gradient_f32,
};

// Input typed array constructors (for copyIn)
type AnyTypedArrayCtor = new (length: number) => TypedArray;
const inCtorMap: Partial<Record<DType, AnyTypedArrayCtor>> = {
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

// Output typed array constructors — f32→f32, everything else→f64
const outCtorMap: Partial<Record<DType, new (length: number) => TypedArray>> = {
  float64: Float64Array,
  float32: Float32Array,
  int64: Float64Array,
  uint64: Float64Array,
  int32: Float64Array,
  uint32: Float64Array,
  int16: Float64Array,
  uint16: Float64Array,
  int8: Float64Array,
  uint8: Float64Array,
  float16: Float32Array,
};

const outDtypeMap: Partial<Record<DType, DType>> = {
  float64: 'float64',
  float32: 'float32',
  int64: 'float64',
  uint64: 'float64',
  int32: 'float64',
  uint32: 'float64',
  int16: 'float64',
  uint16: 'float64',
  int8: 'float64',
  uint8: 'float64',
  float16: 'float32',
};

/**
 * WASM-accelerated 1D gradient with uniform spacing.
 * Returns null if WASM can't handle.
 */
export function wasmGradient1D(a: ArrayStorage, spacing: number): ArrayStorage | null {
  if (!a.isCContiguous) return null;
  if (a.shape.length !== 1) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;
  if (size < 2) return null;

  const dtype = effectiveDType(a.dtype);
  const kernel = kernels[dtype];
  const InCtor = inCtorMap[dtype];
  const OutCtor = outCtorMap[dtype];
  const outDtype = outDtypeMap[dtype];
  if (!kernel || !InCtor || !OutCtor || !outDtype) return null;

  const inBpe = (InCtor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const outBpe = (OutCtor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const outBytes = size * outBpe;

  const outRegion = wasmMalloc(outBytes);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  const aOff = a.offset;
  if (dtype === 'float16') {
    const aPtr = f16InputToScratchF32(a, size);
    kernel(aPtr, outRegion.ptr, size, spacing);
    // Downcast f32 result to f16 if native Float16Array available
    if (hasFloat16) {
      const f16Region = f32OutputToF16Region(outRegion, size);
      outRegion.release();
      if (!f16Region) return null;
      return ArrayStorage.fromWasmRegion(
        [size],
        'float16',
        f16Region,
        size,
        Float16Array as unknown as new (
          buffer: ArrayBuffer,
          byteOffset: number,
          length: number
        ) => TypedArray
      );
    }
  } else {
    const aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, aOff, size, inBpe);
    kernel(aPtr, outRegion.ptr, size, spacing);
  }

  return ArrayStorage.fromWasmRegion(
    [size],
    outDtype,
    outRegion,
    size,
    OutCtor as unknown as new (
      buffer: ArrayBuffer,
      byteOffset: number,
      length: number
    ) => TypedArray
  );
}
